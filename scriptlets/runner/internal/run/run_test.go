package run

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/canonical/starlark/starlark"

	domainunitless "scriptlet-runner/internal/domainunitless"
	"scriptlet-runner/internal/unitless"
)

const testdata = "../../testdata"

func charmDir(t *testing.T, files map[string]string) string {
	t.Helper()
	root := t.TempDir()
	files["charmcraft.yaml"] = "assumes: [unitless]\n"
	for name, content := range files {
		p := filepath.Join(root, filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return root
}

func event(t *testing.T, name, attrs string) Event {
	t.Helper()
	return Event{Name: name, Attrs: json.RawMessage(attrs)}
}

func runEvent(t *testing.T, root string, ev Event, opts Options) *Result {
	t.Helper()
	res := New().Run(context.Background(), root, ev, opts)
	if res.Protocol != ProtocolVersion {
		t.Fatalf("protocol = %d", res.Protocol)
	}
	return res
}

func noMeasure() Options {
	f := false
	return Options{Measure: &f}
}

// --- UseNumber conversion ---------------------------------------------------

func TestDecodeAttrsUseNumber(t *testing.T) {
	got, err := DecodeAttrs(json.RawMessage(`{"i": 8080, "f": 8080.0, "e": 1e3, "neg": -3, "big": 9223372036854775807,
		"nested": {"l": [1, 1.5, "s", true, null]}}`))
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]any{
		"i": int64(8080), "f": float64(8080), "e": float64(1000), "neg": int64(-3),
		"big":    int64(9223372036854775807),
		"nested": map[string]any{"l": []any{int64(1), 1.5, "s", true, nil}},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %#v\nwant %#v", got, want)
	}
	if _, err := DecodeAttrs(json.RawMessage(`{"x": 9223372036854775808}`)); err == nil || !strings.Contains(err.Error(), "out of int64 range") {
		t.Fatalf("overflow err = %v", err)
	}
	if _, err := DecodeAttrs(json.RawMessage(`[1]`)); err == nil {
		t.Fatal("non-object attrs accepted")
	}
}

// An integral float stays a float in Starlark and an int stays an int, as
// with Juju's valueToStarlark (the seed harness turned 8080.0 into 8080).
func TestIntegralFloatVsInt(t *testing.T) {
	root := filepath.Join(testdata, "set_status")
	for _, tc := range []struct {
		attrs, status, message, debug string
	}{
		{`{"port": 8080, "ratio": 1.0}`, "active", "serving on 8080", "port is a int and ratio is a float"},
		{`{"port": 8080.0, "ratio": 1}`, "blocked", "port must be an int, got float", "port is a float and ratio is a int"},
	} {
		res := runEvent(t, root, event(t, "config_changed", tc.attrs), noMeasure())
		if !res.OK {
			t.Fatalf("%s: %+v", tc.attrs, res.Error)
		}
		want := []Intent{{Type: "set-status", Args: map[string]any{"status": tc.status, "message": tc.message}}}
		if !reflect.DeepEqual(res.Intents, want) {
			t.Fatalf("%s: intents = %+v", tc.attrs, res.Intents)
		}
		if res.Logs[0].Message != tc.debug || res.Logs[0].Level != "debug" {
			t.Fatalf("%s: first log = %+v", tc.attrs, res.Logs[0])
		}
	}
}

func TestInvalidInput(t *testing.T) {
	res := runEvent(t, filepath.Join(testdata, "set_status"), event(t, "config_changed", `{"port": 1e400}`), noMeasure())
	if res.Error == nil || res.Error.Kind != "invalid_input" || res.Error.Phase != "input" {
		t.Fatalf("error = %+v", res.Error)
	}
}

// --- set_status intents -----------------------------------------------------

func TestSetStatusIntents(t *testing.T) {
	root := charmDir(t, map[string]string{"scriptlets/hook.star": `
def handler(event):
    juju.set_status("maintenance")
    juju.set_status("active", "ready")
    juju.set_status(status="blocked", message="kw")
    juju.set_status("not-a-real-status", message=event.msg)

def init():
    juju.observe("config_changed", handler)
`})
	res := runEvent(t, root, event(t, "config_changed", `{"msg": "anything goes"}`), Options{})
	if !res.OK {
		t.Fatalf("%+v", res.Error)
	}
	want := []Intent{
		{"set-status", map[string]any{"status": "maintenance", "message": ""}},
		{"set-status", map[string]any{"status": "active", "message": "ready"}},
		{"set-status", map[string]any{"status": "blocked", "message": "kw"}},
		// Juju doesn't validate the status value.
		{"set-status", map[string]any{"status": "not-a-real-status", "message": "anything goes"}},
	}
	if !reflect.DeepEqual(res.Intents, want) {
		t.Fatalf("intents = %+v", res.Intents)
	}
	// JSON shape: {type, args}.
	b, _ := json.Marshal(res.Intents[1])
	if string(b) != `{"type":"set-status","args":{"message":"ready","status":"active"}}` {
		t.Fatalf("json = %s", b)
	}
}

// The protocol's event name is a plain string; Juju's Event.Name is a
// domainunitless.EventType. Juju's own config_changed name must reach the
// scriptlet unchanged: dispatched to the handler observing it, and named in
// the result and in each log entry.
func TestConfigChangedEventName(t *testing.T) {
	name := string(domainunitless.EventConfigChanged)
	if name != "config_changed" {
		t.Fatalf("EventConfigChanged = %q", name)
	}
	root := charmDir(t, map[string]string{"scriptlets/hook.star": `
def on_config(event):
    print("config")
    juju.set_status("active", "config")

def on_other(event):
    juju.set_status("blocked", "other")

def init():
    juju.observe("config_changed", on_config)
    juju.observe("update_status", on_other)
`})
	res := runEvent(t, root, event(t, name, `{}`), noMeasure())
	if !res.OK {
		t.Fatalf("%+v", res.Error)
	}
	if res.Event != name || !res.Observed {
		t.Fatalf("event = %q, observed = %v", res.Event, res.Observed)
	}
	want := []Intent{{"set-status", map[string]any{"status": "active", "message": "config"}}}
	if !reflect.DeepEqual(res.Intents, want) {
		t.Fatalf("intents = %+v", res.Intents)
	}
	if len(res.Logs) != 1 || res.Logs[0].Event != name || res.Logs[0].Message != "config" {
		t.Fatalf("logs = %+v", res.Logs)
	}
}

// Juju's Handle returns no intents when the handler fails, even ones
// appended before the error.
func TestIntentsDroppedOnError(t *testing.T) {
	res := runEvent(t, filepath.Join(testdata, "runtime_error"), event(t, "config_changed", `{"config": {"host": "h"}}`), noMeasure())
	if res.OK || len(res.Intents) != 0 {
		t.Fatalf("ok=%v intents=%+v", res.OK, res.Intents)
	}
	if len(res.Logs) != 1 || res.Logs[0].Message != "config changed" {
		t.Fatalf("logs = %+v", res.Logs) // output before the error is kept
	}
}

func TestSetStatusDuringLoadIsUnavailable(t *testing.T) {
	root := charmDir(t, map[string]string{"scriptlets/hook.star": `
def init():
    juju.set_status("active")
`})
	res := runEvent(t, root, event(t, "config_changed", `{}`), noMeasure())
	if res.Error == nil || res.Error.Kind != "runtime_error" || res.Error.Phase != "load" || !strings.Contains(res.Error.Message, "unavailable") {
		t.Fatalf("error = %+v", res.Error)
	}
}

// --- logs -------------------------------------------------------------------

func TestLogs(t *testing.T) {
	root := charmDir(t, map[string]string{"scriptlets/hook.star": `
print("loading", 1)

def handler(event):
    print("a", "b", sep="-")
    debug({"k": 1})

def init():
    juju.observe("config_changed", handler)
`})
	res := runEvent(t, root, event(t, "config_changed", `{}`), noMeasure())
	if !res.OK {
		t.Fatal(res.Error)
	}
	wantLoad := []LogEntry{{Level: "print", Event: "<load>", Path: "hook.star", Line: 2, Message: "loading 1", Juju: "scriptlet hook.star:2: loading 1", JujuLevel: "INFO"}}
	if !reflect.DeepEqual(res.Load.Logs, wantLoad) {
		t.Fatalf("load logs = %+v", res.Load.Logs)
	}
	want := []LogEntry{
		{Level: "print", Event: "config_changed", Path: "hook.star", Line: 5, Message: "a-b", Juju: "scriptlet hook.star:5: a-b", JujuLevel: "INFO"},
		{Level: "debug", Event: "config_changed", Path: "hook.star", Line: 6, Message: `{"k": 1}`, Juju: `scriptlet config_changed: hook.star:6: {"k": 1}`, JujuLevel: "DEBUG"},
	}
	if !reflect.DeepEqual(res.Logs, want) {
		t.Fatalf("logs = %+v", res.Logs)
	}
}

// --- error kinds ------------------------------------------------------------

func TestErrorKinds(t *testing.T) {
	handler := func(body string) map[string]string {
		return map[string]string{"scriptlets/hook.star": "def handler(event):\n" + body + "\n\ndef init():\n    juju.observe(\"config_changed\", handler)\n"}
	}
	for _, tc := range []struct {
		name    string
		files   map[string]string
		ev      string
		opts    Options
		kind    string
		phase   string
		message string
		pos     *Position
		builtin string
	}{
		{name: "parse error", files: map[string]string{"scriptlets/hook.star": "def f(:\n"},
			kind: "syntax_error", phase: "load", message: "hook.star:1:8: got ':', want ')'", pos: &Position{"hook.star", 1, 8}},
		{name: "dialect (while)", files: map[string]string{"scriptlets/hook.star": "def f():\n    while True:\n        pass\n"},
			kind: "syntax_error", phase: "load", message: "while loops", pos: &Position{"hook.star", 2, 5}},
		{name: "undefined name", files: map[string]string{"scriptlets/aaa.star": "x = 1\n", "scriptlets/hook.star": "x = struct(a=1)\n"},
			kind: "syntax_error", phase: "load", message: "hook.star:1:5: undefined: struct", pos: &Position{"hook.star", 1, 5}},
		{name: "runtime", files: handler("    x = {}[\"k\"]"),
			kind: "runtime_error", phase: "event", message: `key "k" not in dict`, pos: &Position{"hook.star", 2, 11}},
		{name: "builtin non-argument error is runtime", files: handler("    x = len(1)"),
			kind: "runtime_error", phase: "event", message: "len: value of type int has no len", builtin: "len"},
		{name: "runtime at load", files: map[string]string{"scriptlets/hook.star": "x = 1 // 0\n"},
			kind: "runtime_error", phase: "load", message: "floored division by zero"},
		{name: "step limit", files: handler("    x = [i for i in range(event.n)]"), ev: `{"n": 100000}`,
			kind: "step_limit", phase: "event", message: "too many steps"},
		{name: "step limit at load", files: map[string]string{"scriptlets/hook.star": "def init():\n    x = [i for i in range(200000)]\n"},
			kind: "step_limit", phase: "load", message: "too many steps"},
		{name: "alloc limit", files: handler("    x = [None] * 1000"), opts: Options{MaxAllocs: 4096},
			kind: "alloc_limit", phase: "event", message: "exceeded memory allocation limits"},
		{name: "set_status missing argument", files: handler("    juju.set_status()"),
			kind: "builtin_argument_error", phase: "event", message: "set_status: missing argument for status", builtin: "set_status"},
		{name: "set_status wrong type", files: handler("    juju.set_status(1)"),
			kind: "builtin_argument_error", phase: "event", message: "set_status: for parameter status: got int, want string", builtin: "set_status"},
		{name: "set_status unknown keyword", files: handler("    juju.set_status('active', colour='green')"),
			kind: "builtin_argument_error", phase: "event", message: "set_status: unexpected keyword argument \"colour\"", builtin: "set_status"},
		{name: "observe outside init", files: handler("    juju.observe('x_y', handler)"),
			kind: "runtime_error", phase: "event", message: "unavailable"},
		{name: "invalid file name", files: map[string]string{"scriptlets/my-hook.star": "x = 1\n"},
			kind: "load_path_error", phase: "load", message: `cannot load my-hook.star: path contains "-", use "_" instead`},
		{name: "short file name", files: map[string]string{"scriptlets/ab.star": "x = 1\n"},
			kind: "load_path_error", phase: "load", message: "path invalid"},
		{name: "load json module", files: map[string]string{"scriptlets/hook.star": "load(\"json\", \"decode\")\n"},
			kind: "load_path_error", phase: "load", message: "cannot load json: path invalid"},
		{name: "load missing file", files: map[string]string{"scriptlets/hook.star": "load(\"lib/nope.star\", \"x\")\n"},
			kind: "load_path_error", phase: "load", message: "cannot load lib/nope.star: file not found"},
		{name: "empty source", files: map[string]string{"scriptlets/hook.star": "x = 1\n", "scriptlets/empty.star": ""},
			kind: "charm_invalid", phase: "load", message: `empty scriptlet source "empty.star" not valid`},
		{name: "invalid event name", files: handler("    pass"), ev: "", kind: "invalid_event", phase: "event", message: "name too short"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			root := charmDir(t, tc.files)
			name := "config_changed"
			if tc.kind == "invalid_event" {
				name = "cc"
			}
			attrs := tc.ev
			if attrs == "" {
				attrs = "{}"
			}
			opts := tc.opts
			f := false
			opts.Measure = &f
			res := runEvent(t, root, event(t, name, attrs), opts)
			e := res.Error
			if res.OK || e == nil {
				t.Fatalf("no error: %+v", res)
			}
			if e.Kind != tc.kind || e.Phase != tc.phase || !strings.Contains(e.Message, tc.message) {
				t.Fatalf("got %s/%s %q, want %s/%s %q (juju_log %q)", e.Kind, e.Phase, e.Message, tc.kind, tc.phase, tc.message, e.JujuLog)
			}
			if tc.pos != nil && !reflect.DeepEqual(e.Position, tc.pos) {
				t.Fatalf("position = %+v, want %+v", e.Position, tc.pos)
			}
			if e.Builtin != tc.builtin {
				t.Fatalf("builtin = %q, want %q", e.Builtin, tc.builtin)
			}
			if strings.Contains(e.Message, "cached-") || strings.Contains(e.Backtrace, "cached-") {
				t.Fatalf("unmapped file name: %q %q", e.Message, e.Backtrace)
			}
		})
	}
}

func TestRuntimeErrorBacktrace(t *testing.T) {
	res := runEvent(t, filepath.Join(testdata, "runtime_error"), event(t, "config_changed", `{"config": {"host": "h"}}`), noMeasure())
	want := "Traceback (most recent call last):\n  hook.star:7:39: in on_config_changed\n  hook.star:2:52: in endpoint\nError: key \"port\" not in dict"
	if res.Error.Backtrace != want {
		t.Fatalf("backtrace = %q", res.Error.Backtrace)
	}
	if res.Error.JujuLog != `handling event "config_changed": key "port" not in dict` {
		t.Fatalf("juju_log = %q", res.Error.JujuLog)
	}
}

// --- measurement --------------------------------------------------------

// The defining property of Used: the event passes at Used and fails on that
// budget at Used-1.
func TestMeasuredIsMinimum(t *testing.T) {
	root := filepath.Join(testdata, "step_limit")
	raw, err := os.ReadFile(filepath.Join(testdata, "events/relation_changed_100.json"))
	if err != nil {
		t.Fatal(err)
	}
	var ev Event
	json.Unmarshal(raw, &ev)
	res := runEvent(t, root, ev, Options{})
	if !res.OK || res.Steps.Used == nil || res.Allocs.Used == nil || res.Steps.LimitHit {
		t.Fatalf("%+v", res)
	}
	steps, allocs := *res.Steps.Used, *res.Allocs.Used
	check := func(o Options, wantOK bool, kind string) {
		t.Helper()
		f := false
		o.Measure = &f
		r := runEvent(t, root, ev, o)
		if r.OK != wantOK || (!wantOK && r.Error.Kind != kind) {
			t.Fatalf("opts %+v: ok=%v err=%+v", o, r.OK, r.Error)
		}
	}
	check(Options{MaxSteps: steps}, true, "")
	check(Options{MaxSteps: steps - 1}, false, "step_limit")
	check(Options{MaxAllocs: allocs}, true, "")
	check(Options{MaxAllocs: allocs - 1}, false, "alloc_limit")

	// Same for the load phase.
	ls, la := *res.Load.Steps.Used, *res.Load.Allocs.Used
	// (The handler needs more than the load, so only the load phase is
	// checked here.)
	for _, o := range []struct {
		opts Options
		fail string
	}{
		{Options{MaxSteps: ls}, ""}, {Options{MaxSteps: ls - 1}, "step_limit"},
		{Options{MaxAllocs: la}, ""}, {Options{MaxAllocs: la - 1}, "alloc_limit"},
	} {
		o.opts.Measure = new(bool)
		r := runEvent(t, root, ev, o.opts)
		loadFailed := r.Error != nil && r.Error.Phase == "load"
		if (o.fail == "") == loadFailed || (loadFailed && r.Error.Kind != o.fail) {
			t.Fatalf("load with %+v: %+v", o.opts, r.Error)
		}
	}
}

func TestLimitHitReportsNeed(t *testing.T) {
	raw, _ := os.ReadFile(filepath.Join(testdata, "events/relation_changed_2000.json"))
	var ev Event
	json.Unmarshal(raw, &ev)
	res := runEvent(t, filepath.Join(testdata, "step_limit"), ev, Options{})
	if res.OK || res.Error.Kind != "step_limit" || !res.Steps.LimitHit || res.Steps.Cap != 1_000_000 {
		t.Fatalf("%+v %+v", res.Error, res.Steps)
	}
	if res.Steps.Used == nil || *res.Steps.Used <= 100_000 {
		t.Fatalf("used = %v", res.Steps.Used)
	}
	// Past the cap: Used stays null.
	res = runEvent(t, filepath.Join(testdata, "step_limit"), ev, Options{CapFactor: 1})
	if res.Steps.Used != nil {
		t.Fatalf("used = %d with cap = limit", *res.Steps.Used)
	}
}

func TestUnobservedEvent(t *testing.T) {
	res := runEvent(t, filepath.Join(testdata, "set_status"), event(t, "relation_broken", ""), Options{})
	if !res.OK || res.Observed || len(res.Intents) != 0 || *res.Steps.Used != 0 || *res.Allocs.Used != 0 {
		t.Fatalf("%+v", res)
	}
}

// starlark 8c33ed2's Call leaves a half-pushed frame on the stack when the
// allocation for growing thread.stack fails, and the caller's evalError then
// dereferences its nil callable. The runner must report it, not crash.
func TestInterpreterPanicRecovered(t *testing.T) {
	root := charmDir(t, map[string]string{"scriptlets/hook.star": `
def r(d):
    return 0 if d == 0 else r(d - 1)

def handler(event):
    x = "a" * 1000
    r(40)

def init():
    juju.observe("config_changed", handler)
`})
	r := New()
	f := false
	panics := 0
	for lim := int64(2000); lim < 6000; lim += 7 {
		res := r.Run(context.Background(), root, event(t, "config_changed", "{}"), Options{MaxAllocs: lim, Measure: &f})
		if res.Error != nil && res.Error.Kind == "interpreter_panic" {
			panics++
			if !strings.Contains(res.Error.Message, "nil pointer dereference") {
				t.Fatalf("message = %q", res.Error.Message)
			}
		}
	}
	if panics == 0 {
		t.Fatal("no limit in range panicked; has starlark fixed the stack-growth bug?")
	}
	t.Logf("%d limits panicked", panics)
}

// --- fidelity to Juju's own executor -----------------------------------

// The ProbeExecutor must behave like Juju's NewStarformExecutor.
func TestProbeExecutorMatchesJujuExecutor(t *testing.T) {
	root := filepath.Join(testdata, "set_status")
	srcs := []domainunitless.ScriptSource{}
	for _, p := range []string{"charm.star", "lib/checks.star"} {
		b, _ := os.ReadFile(filepath.Join(root, "scriptlets", p))
		srcs = append(srcs, domainunitless.ScriptSource{LoadPath: p, Source: string(b)})
	}
	cfg := unitless.ExecutorConfig{Scriptlet: domainunitless.ScriptletApplication{Sources: srcs}, MaxSteps: 100_000, MaxAllocs: 10 << 20, Logger: &captureLogger{}}
	ctx := context.Background()
	juju, err := unitless.NewStarformExecutor(ctx, cfg)
	if err != nil {
		t.Fatal(err)
	}
	probe, err := unitless.NewProbeExecutor(ctx, cfg)
	if err != nil {
		t.Fatal(err)
	}
	for _, attrs := range []map[string]any{{"port": int64(80), "ratio": 0.5}, {"port": 80.0, "ratio": 0.5}, {"port": int64(0), "ratio": 0.5}, {}} {
		ev := domainunitless.Event{Name: "config_changed", Attrs: attrs}
		a, aerr := juju.Handle(ctx, ev)
		b, berr := probe.Handle(ctx, ev, 100_000, 10<<20)
		if !reflect.DeepEqual(a, b) || (aerr == nil) != (berr == nil) || (aerr != nil && aerr.Error() != berr.Error()) {
			t.Fatalf("attrs %v: juju %v %v, probe %v %v", attrs, a, aerr, b, berr)
		}
	}
}

var _ = starlark.None

// --- load op ----------------------------------------------------------------

func TestLoadWithoutEvent(t *testing.T) {
	root := charmDir(t, map[string]string{"scriptlets/hook.star": `
print("top level")

def handler(event):
    print("handled")

def init():
    debug("init")
    juju.observe("config_changed", handler)
`})
	r := New()
	res := r.Load(context.Background(), root, Options{})
	if res.Protocol != ProtocolVersion || !res.OK || res.Error != nil {
		t.Fatalf("load = %+v", res)
	}
	if res.Load.Cached || !reflect.DeepEqual(res.Load.Sources, []string{"hook.star"}) {
		t.Fatalf("load = %+v", res.Load)
	}
	var msgs []string
	for _, e := range res.Load.Logs {
		msgs = append(msgs, e.Event+":"+e.Message)
	}
	if !reflect.DeepEqual(msgs, []string{"<load>:top level", "<load>:init"}) {
		t.Fatalf("load logs = %v", msgs)
	}
	if !res.Load.Steps.Measured || res.Load.Steps.Used == nil || *res.Load.Steps.Used == 0 || res.Load.Allocs.Used == nil {
		t.Fatalf("load not measured: %+v %+v", res.Load.Steps, res.Load.Allocs)
	}
	// The load matches what a run reports, and the run reuses it.
	ev := r.Run(context.Background(), root, event(t, "config_changed", `{}`), Options{})
	if !ev.OK || !ev.Load.Cached || *ev.Load.Steps.Used != *res.Load.Steps.Used || len(ev.Logs) != 1 {
		t.Fatalf("run after load = %+v / %+v", ev, ev.Load)
	}
	// Unmeasured: same load result shape, nothing measured.
	un := New().Load(context.Background(), root, noMeasure())
	if !un.OK || un.Load.Steps.Measured || un.Load.Steps.Used != nil || un.Load.Steps.Limit != unitless.DefaultMaxSteps {
		t.Fatalf("unmeasured load = %+v", un.Load)
	}
}

func TestLoadErrors(t *testing.T) {
	for _, tc := range []struct {
		name, phase, kind string
		files             map[string]string
		opts              Options
	}{
		{"syntax", "load", "syntax_error", map[string]string{"scriptlets/hook.star": "def init(:\n"}, noMeasure()},
		{"runtime", "load", "runtime_error", map[string]string{"scriptlets/hook.star": "def init():\n    {}['x']\n"}, noMeasure()},
		{"steps", "load", "step_limit", map[string]string{"scriptlets/hook.star": "def init():\n    for i in range(1000):\n        pass\n"}, Options{MaxSteps: 50}},
		{"empty source", "load", "charm_invalid", map[string]string{"scriptlets/hook.star": ""}, noMeasure()},
		{"hooks", "charm", "charm_invalid", map[string]string{"scriptlets/hook.star": "def init():\n    pass\n", "hooks/install": "#!/bin/sh\n"}, noMeasure()},
	} {
		t.Run(tc.name, func(t *testing.T) {
			res := New().Load(context.Background(), charmDir(t, tc.files), tc.opts)
			if res.OK || res.Error == nil || res.Error.Kind != tc.kind || res.Error.Phase != tc.phase {
				t.Fatalf("load = %+v, error = %+v", res, res.Error)
			}
			if tc.phase == "load" && res.Load == nil {
				t.Fatal("no load result for a load-phase error")
			}
		})
	}
	// A step-limit load is measured past the limit, as for a run.
	res := New().Load(context.Background(), charmDir(t, map[string]string{"scriptlets/hook.star": "def init():\n    for i in range(100):\n        pass\n"}), Options{MaxSteps: 50})
	if !res.Load.Steps.LimitHit || res.Load.Steps.Used == nil || *res.Load.Steps.Used <= 50 {
		t.Fatalf("steps = %+v", res.Load.Steps)
	}
	res = New().Load(context.Background(), filepath.Join(t.TempDir(), "missing"), Options{})
	if res.Error == nil || res.Error.Kind != "invalid_input" || res.Error.Phase != "charm" {
		t.Fatalf("missing charm = %+v", res.Error)
	}
}
