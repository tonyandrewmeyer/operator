// Package run loads a unitless charm's scriptlets and runs one event through
// Juju's own executor code (copied into internal/unitless), returning the
// intents, the print/debug output, the steps and allocations the load and
// the event used, and a structured error.
package run

import (
	"bytes"
	"context"
	"crypto/sha256"
	"crypto/sha512"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
	"strings"

	"github.com/canonical/starform/starform"

	"scriptlet-runner/internal/charm"
	domainunitless "scriptlet-runner/internal/domainunitless"
	"scriptlet-runner/internal/unitless"
)

// ProtocolVersion is bumped on any incompatible change to Result,
// LoadResponse or the serve requests.
const ProtocolVersion = 1

// DefaultCapFactor bounds how far past the real limit the runner searches
// when the real limit was hit (to say how much a failing event needed).
const DefaultCapFactor = 10

// Options control one run.
type Options struct {
	// MaxSteps and MaxAllocs default to Juju's (100,000 and 10 MiB).
	MaxSteps  int64 `json:"max_steps,omitempty"`
	MaxAllocs int64 `json:"max_allocs,omitempty"`
	// Measure bisects for the steps and allocations used. Defaults to true
	// (a nil pointer); set false to skip it and run each phase once.
	Measure *bool `json:"measure,omitempty"`
	// CapFactor: when the real limit is hit, search up to CapFactor x limit.
	CapFactor int64 `json:"cap_factor,omitempty"`
	// SkipCharmChecks skips Juju's add-charm checks (assumes: [unitless],
	// no hooks or dispatch alongside scriptlets).
	SkipCharmChecks bool `json:"skip_charm_checks,omitempty"`
}

func (o Options) withDefaults() Options {
	if o.MaxSteps == 0 {
		o.MaxSteps = unitless.DefaultMaxSteps
	}
	if o.MaxAllocs == 0 {
		o.MaxAllocs = unitless.DefaultMaxAllocs
	}
	if o.CapFactor == 0 {
		o.CapFactor = DefaultCapFactor
	}
	return o
}

func (o Options) measure() bool { return o.Measure == nil || *o.Measure }

// Event is the event to run: a name and attributes, as JSON.
type Event struct {
	Name  string          `json:"name"`
	Attrs json.RawMessage `json:"attrs,omitempty"`
}

// Intent is Juju's unitless.Intent, as JSON.
type Intent struct {
	Type string         `json:"type"`
	Args map[string]any `json:"args"`
}

// LogEntry is one print or debug call.
type LogEntry struct {
	Level   string `json:"level"` // "print" or "debug"
	Event   string `json:"event"` // "<load>" during loading
	Path    string `json:"path"`
	Line    int    `json:"line"`
	Message string `json:"message"`
	// Juju is the line Juju's starformLogAdapter writes, at JujuLevel, on the
	// juju.worker.unitless logger.
	Juju      string `json:"juju"`
	JujuLevel string `json:"juju_level"`
}

// Budget reports one limit.
type Budget struct {
	Limit int64 `json:"limit"`
	// Used is the smallest limit at which the phase doesn't fail on this
	// budget: the steps or bytes it used. Null when not measured, or when
	// the phase still failed on this budget at Cap.
	Used     *int64 `json:"used"`
	LimitHit bool   `json:"limit_hit"`
	Measured bool   `json:"measured"`
	// Cap is set when the limit was hit and the runner searched above it.
	Cap int64 `json:"cap,omitempty"`
}

// Error is a structured scriptlet or runner error.
type Error struct {
	// Kind is one of: syntax_error, runtime_error, builtin_argument_error,
	// step_limit, alloc_limit, load_path_error, invalid_event,
	// charm_invalid, invalid_input, interpreter_panic, internal.
	Kind  string `json:"kind"`
	Phase string `json:"phase"` // "input", "charm", "load" or "event"
	// Message is the innermost starform/Starlark message, with starform's
	// cached-<hash>.star file names mapped back to load paths.
	Message string `json:"message"`
	// JujuLog is the error text as Juju would log it (unmapped file names,
	// with the executor's wrapping).
	JujuLog   string    `json:"juju_log,omitempty"`
	Backtrace string    `json:"backtrace,omitempty"`
	Position  *Position `json:"position,omitempty"`
	// Builtin names the builtin the error was raised in, if any.
	Builtin string `json:"builtin,omitempty"`
}

func (e *Error) Error() string { return e.Kind + ": " + e.Message }

// Position is a location in a scriptlet source.
type Position struct {
	Path string `json:"path"`
	Line int32  `json:"line"`
	Col  int32  `json:"col"`
}

// LoadResult describes loading the sources and running every init().
type LoadResult struct {
	Sources []string   `json:"sources"`
	Logs    []LogEntry `json:"logs"`
	Steps   Budget     `json:"steps"`
	Allocs  Budget     `json:"allocs"`
	Cached  bool       `json:"cached"`
}

// Result is the runner's output for one event.
type Result struct {
	Protocol int             `json:"protocol"`
	ID       json.RawMessage `json:"id,omitempty"`
	OK       bool            `json:"ok"`
	Event    string          `json:"event"`
	Observed bool            `json:"observed"`
	Intents  []Intent        `json:"intents"`
	Logs     []LogEntry      `json:"logs"`
	Steps    *Budget         `json:"steps"`
	Allocs   *Budget         `json:"allocs"`
	Load     *LoadResult     `json:"load"`
	Error    *Error          `json:"error"`
}

// captureLogger is the starform Logger handed to the script set. starform
// binds print and debug to it at load time, so it stays the same object and
// its sink is switched between phases (and off during probes).
type captureLogger struct {
	sink *[]LogEntry
}

func (l *captureLogger) Log(_ context.Context, e starform.LogEntry) {
	if l.sink == nil {
		return
	}
	level, jujuLevel := "print", "INFO"
	if e.Level == starform.DebugLevel {
		level, jujuLevel = "debug", "DEBUG"
	}
	*l.sink = append(*l.sink, LogEntry{
		Level: level, Event: e.EventName, Path: e.Path, Line: e.Line, Message: e.Message,
		Juju: "scriptlet " + e.String(), JujuLevel: jujuLevel,
	})
}

// loaded is a loaded charm, reusable across events (Juju also loads once per
// application and handles every event on the same script set).
type loaded struct {
	exec    *unitless.ProbeExecutor
	logger  *captureLogger
	load    LoadResult
	err     *Error
	names   map[string]string // cached-<hash>.star -> load path
	sources []domainunitless.ScriptSource
}

// Runner runs events. It caches loaded charms by content and limits, which
// only matters in serve mode.
type Runner struct {
	cache map[[32]byte]*loaded
}

// New returns a Runner.
func New() *Runner { return &Runner{cache: map[[32]byte]*loaded{}} }

// Run loads the charm at root (directory or .charm file) and runs event.
func (r *Runner) Run(ctx context.Context, root string, event Event, opts Options) *Result {
	opts = opts.withDefaults()
	res := &Result{Protocol: ProtocolVersion, Event: event.Name, Intents: []Intent{}, Logs: []LogEntry{}}

	attrs, err := DecodeAttrs(event.Attrs)
	if err != nil {
		res.Error = &Error{Kind: "invalid_input", Phase: "input", Message: err.Error()}
		return res
	}

	l, load, cerr := r.loadCharm(ctx, root, opts)
	res.Load = load
	if cerr != nil {
		res.Error = cerr
		return res
	}

	res.Observed = l.exec.IsObserved(event.Name)
	ev := domainunitless.Event{Name: domainunitless.EventType(event.Name), Attrs: attrs}

	var logs []LogEntry
	l.logger.sink = &logs
	var intents []unitless.Intent
	herr := guard(func() (err error) {
		intents, err = l.exec.Handle(ctx, ev, opts.MaxSteps, opts.MaxAllocs)
		return err
	})
	l.logger.sink = nil
	if logs != nil {
		res.Logs = logs
	}
	for _, in := range intents {
		res.Intents = append(res.Intents, Intent{Type: string(in.Type), Args: in.Args})
	}
	res.Steps = &Budget{Limit: opts.MaxSteps, LimitHit: isSteps(herr)}
	res.Allocs = &Budget{Limit: opts.MaxAllocs, LimitHit: isAllocs(herr)}
	if herr != nil {
		res.Error = classify(herr, "event", l.names)
	}
	res.OK = herr == nil

	if opts.measure() && !res.Observed && herr == nil {
		// starform's Handle returns before making a thread: nothing ran.
		zero := int64(0)
		res.Steps.Used, res.Steps.Measured = &zero, true
		res.Allocs.Used, res.Allocs.Measured = &zero, true
	} else if opts.measure() {
		run := func(steps, allocs int64) error {
			return guard(func() error {
				_, err := l.exec.Handle(ctx, ev, steps, allocs)
				return err
			})
		}
		capSteps, capAllocs := opts.MaxSteps*opts.CapFactor, opts.MaxAllocs*opts.CapFactor
		measure(res.Steps, herr, isSteps, capSteps, func(n int64) error { return run(n, capAllocs) })
		measure(res.Allocs, herr, isAllocsOrPanic, capAllocs, func(n int64) error { return run(capSteps, n) })
	}
	return res
}

// LoadResponse is the runner's output for the load op: the charm's sources
// loaded and every init() run, with no event.
type LoadResponse struct {
	Protocol int             `json:"protocol"`
	ID       json.RawMessage `json:"id,omitempty"`
	OK       bool            `json:"ok"`
	Load     *LoadResult     `json:"load"`
	Error    *Error          `json:"error"`
}

// Load loads the charm at root (directory or .charm file) without running an
// event. The loaded script set is cached as for Run, so a following Run with
// the same sources and options reuses it.
func (r *Runner) Load(ctx context.Context, root string, opts Options) *LoadResponse {
	opts = opts.withDefaults()
	_, load, err := r.loadCharm(ctx, root, opts)
	return &LoadResponse{Protocol: ProtocolVersion, OK: err == nil, Load: load, Error: err}
}

// loadCharm reads the charm and loads its script set (or takes it from the
// cache). The LoadResult is a copy, nil when the charm itself couldn't be
// read. opts must already have its defaults.
func (r *Runner) loadCharm(ctx context.Context, root string, opts Options) (*loaded, *LoadResult, *Error) {
	ch, err := charm.Load(root, !opts.SkipCharmChecks)
	if err != nil {
		var inv *charm.InvalidError
		kind := "invalid_input"
		if errors.As(err, &inv) {
			kind = "charm_invalid"
		}
		return nil, nil, &Error{Kind: kind, Phase: "charm", Message: err.Error()}
	}
	l := r.load(ctx, ch, opts)
	load := l.load
	return l, &load, l.err
}

func (r *Runner) load(ctx context.Context, ch *charm.Charm, opts Options) *loaded {
	sources := make([]domainunitless.ScriptSource, len(ch.Sources))
	names := map[string]string{}
	h := sha256.New()
	for i, s := range ch.Sources {
		sources[i] = domainunitless.ScriptSource{LoadPath: s.Path, Source: string(s.Content)}
		key := sha512.Sum384(s.Content)
		names[fmt.Sprintf("cached-%x.star", key[:4])] = s.Path
		fmt.Fprintf(h, "%d:%s%d:", len(s.Path), s.Path, len(s.Content))
		h.Write(s.Content)
	}
	binary.Write(h, binary.LittleEndian, [3]int64{opts.MaxSteps, opts.MaxAllocs, opts.CapFactor})
	var b2 bool = opts.measure()
	binary.Write(h, binary.LittleEndian, b2)
	var key [32]byte
	copy(key[:], h.Sum(nil))
	if l, ok := r.cache[key]; ok {
		l.load.Cached = true
		return l
	}

	l := &loaded{logger: &captureLogger{}, names: names, sources: sources}
	l.load = LoadResult{Sources: make([]string, len(sources)), Logs: []LogEntry{}}
	for i, s := range sources {
		l.load.Sources[i] = s.LoadPath
	}
	cfg := func(steps, allocs int64, logger starform.Logger) unitless.ExecutorConfig {
		return unitless.ExecutorConfig{
			Scriptlet: domainunitless.ScriptletApplication{Sources: sources},
			MaxSteps:  steps, MaxAllocs: allocs, Logger: logger,
		}
	}
	var logs []LogEntry
	l.logger.sink = &logs
	var exec *unitless.ProbeExecutor
	err := guard(func() (err error) {
		exec, err = unitless.NewProbeExecutor(ctx, cfg(opts.MaxSteps, opts.MaxAllocs, l.logger))
		return err
	})
	l.logger.sink = nil
	if logs != nil {
		l.load.Logs = logs
	}
	l.exec = exec
	l.load.Steps = Budget{Limit: opts.MaxSteps, LimitHit: isSteps(err)}
	l.load.Allocs = Budget{Limit: opts.MaxAllocs, LimitHit: isAllocs(err)}
	if err != nil {
		l.err = classify(err, "load", names)
		// Juju's NewStarformExecutor wraps LoadSources errors like this.
		if !isValidateErr(err) {
			l.err.JujuLog = "loading Starform script sources: " + err.Error()
		}
	}
	if opts.measure() && !isValidateErr(err) {
		// Probe the load with Juju's own constructor; each probe is a fresh
		// script set, as each Juju runner start is.
		discard := &captureLogger{}
		probe := func(steps, allocs int64) error {
			return guard(func() error {
				_, err := unitless.NewStarformExecutor(ctx, cfg(steps, allocs, discard))
				return err
			})
		}
		capSteps, capAllocs := opts.MaxSteps*opts.CapFactor, opts.MaxAllocs*opts.CapFactor
		measure(&l.load.Steps, err, isSteps, capSteps, func(n int64) error { return probe(n, capAllocs) })
		measure(&l.load.Allocs, err, isAllocsOrPanic, capAllocs, func(n int64) error { return probe(capSteps, n) })
	}
	r.cache[key] = l
	return l
}

// measure finds the smallest limit at which run doesn't fail on this
// budget. Runs are deterministic, so the predicate is monotonic in the
// limit. realErr is the outcome at b.Limit.
func measure(b *Budget, realErr error, hit func(error) bool, cap int64, run func(int64) error) {
	b.Measured = true
	pass := func(n int64) bool { return !hit(run(n)) }
	lo, hi := int64(1), b.Limit
	if hit(realErr) {
		b.Cap = cap
		if !pass(cap) {
			return // still failing at the cap: Used stays null
		}
		lo, hi = b.Limit+1, cap
	}
	for lo < hi {
		mid := lo + (hi-lo)/2
		if pass(mid) {
			hi = mid
		} else {
			lo = mid + 1
		}
	}
	b.Used = &lo
}

// DecodeAttrs decodes event attributes with UseNumber, so that JSON integers
// become int64 and anything with a fraction or exponent becomes float64.
// Juju's valueToStarlark then makes Int from int64 and Float from float64:
// 1 and 1.0 stay distinct, as they would in Juju.
func DecodeAttrs(raw json.RawMessage) (map[string]any, error) {
	if len(bytes.TrimSpace(raw)) == 0 || string(bytes.TrimSpace(raw)) == "null" {
		return map[string]any{}, nil
	}
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()
	var v any
	if err := dec.Decode(&v); err != nil {
		return nil, fmt.Errorf("attrs: %w", err)
	}
	m, ok := v.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("attrs: must be a JSON object, got %T", v)
	}
	out, err := numbers(m)
	if err != nil {
		return nil, fmt.Errorf("attrs: %w", err)
	}
	return out.(map[string]any), nil
}

func numbers(v any) (any, error) {
	switch v := v.(type) {
	case json.Number:
		s := string(v)
		if !strings.ContainsAny(s, ".eE") {
			i, err := strconv.ParseInt(s, 10, 64)
			if err != nil {
				return nil, fmt.Errorf("integer %s is out of int64 range", s)
			}
			return i, nil
		}
		f, err := strconv.ParseFloat(s, 64)
		if err != nil {
			return nil, fmt.Errorf("number %s: %w", s, err)
		}
		return f, nil
	case map[string]any:
		for k, x := range v {
			y, err := numbers(x)
			if err != nil {
				return nil, fmt.Errorf("%q: %w", k, err)
			}
			v[k] = y
		}
		return v, nil
	case []any:
		for i, x := range v {
			y, err := numbers(x)
			if err != nil {
				return nil, fmt.Errorf("[%d]: %w", i, err)
			}
			v[i] = y
		}
		return v, nil
	}
	return v, nil
}
