package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"io"
	"strings"
	"testing"
)

func TestServeRoundTrip(t *testing.T) {
	in := strings.Join([]string{
		`{"id": 1, "op": "version"}`,
		`{"id": "l", "op": "load", "charm": "testdata/set_status"}`,
		`{"id": "a", "charm": "testdata/set_status", "event": {"name": "config_changed", "attrs": {"port": 8080, "ratio": 0.5}}}`,
		``,
		`{"id": "b", "op": "run", "charm": "testdata/set_status", "event": {"name": "config_changed", "attrs": {"port": 0, "ratio": 0.5}}}`,
		`{"id": "c", "charm": "testdata/step_limit", "event": {"name": "relation_changed", "attrs": {"units": []}}, "options": {"max_steps": 10, "measure": false}}`,
		`not json`,
		`{"id": 9, "op": "frobnicate"}`,
		`{"id": 10, "op": "shutdown"}`,
		`{"id": 11, "op": "version"}`,
	}, "\n") + "\n"
	var out bytes.Buffer
	if err := serve(strings.NewReader(in), &out); err != nil {
		t.Fatal(err)
	}
	var lines []map[string]any
	sc := bufio.NewScanner(&out)
	for sc.Scan() {
		var m map[string]any
		if err := json.Unmarshal(sc.Bytes(), &m); err != nil {
			t.Fatalf("bad line %q: %v", sc.Text(), err)
		}
		lines = append(lines, m)
	}
	if len(lines) != 8 { // blank line skipped; nothing after shutdown
		t.Fatalf("%d responses: %v", len(lines), lines)
	}
	if lines[0]["id"] != 1.0 || lines[0]["starform"] == "" || lines[0]["juju_commit"] != JujuCommit || lines[0]["juju_version"] != JujuVersion {
		t.Fatalf("version: %v", lines[0])
	}
	if tag, ok := lines[0]["juju_tag"]; !ok || (JujuTag == "" && tag != nil) || (JujuTag != "" && tag != JujuTag) {
		t.Fatalf("version juju_tag: %v", lines[0])
	}
	ld := lines[1]
	if ld["id"] != "l" || ld["ok"] != true || ld["error"] != nil || ld["protocol"] != 1.0 || ld["load"].(map[string]any)["cached"] != false {
		t.Fatalf("load: %v", ld)
	}
	if _, ok := ld["event"]; ok {
		t.Fatalf("load response has an event: %v", ld)
	}
	lines = append(lines[:1], lines[2:]...)
	intent := func(m map[string]any) map[string]any {
		return m["intents"].([]any)[0].(map[string]any)["args"].(map[string]any)
	}
	if lines[1]["id"] != "a" || lines[1]["ok"] != true || intent(lines[1])["message"] != "serving on 8080" {
		t.Fatalf("run a: %v", lines[1])
	}
	if lines[1]["load"].(map[string]any)["cached"] != true || lines[2]["load"].(map[string]any)["cached"] != true {
		t.Fatal("runs after a load of the same charm should reuse the loaded script set")
	}
	if lines[2]["id"] != "b" || intent(lines[2])["status"] != "blocked" {
		t.Fatalf("run b: %v", lines[2])
	}
	if e := lines[3]["error"].(map[string]any); lines[3]["id"] != "c" || e["kind"] != "step_limit" {
		t.Fatalf("run c: %v", lines[3])
	}
	if e := lines[4]["error"].(map[string]any); e["kind"] != "invalid_input" {
		t.Fatalf("bad json: %v", lines[4])
	}
	if e := lines[5]["error"].(map[string]any); lines[5]["id"] != 9.0 || !strings.Contains(e["message"].(string), "unknown op") {
		t.Fatalf("unknown op: %v", lines[5])
	}
	if lines[6]["id"] != 10.0 || lines[6]["ok"] != true {
		t.Fatalf("shutdown: %v", lines[6])
	}
}

// Responses are flushed per request, so a client can interleave writes and
// reads on pipes.
func TestServeFlushesPerRequest(t *testing.T) {
	inR, inW := io.Pipe()
	outR, outW := io.Pipe()
	done := make(chan error, 1)
	go func() { done <- serve(inR, outW); outW.Close() }()
	rd := bufio.NewReader(outR)
	for i := 0; i < 3; i++ {
		io.WriteString(inW, `{"id": 1, "charm": "testdata/set_status", "event": {"name": "update_status"}}`+"\n")
		line, err := rd.ReadString('\n')
		if err != nil || !strings.Contains(line, `"ok":true`) {
			t.Fatalf("%d: %q %v", i, line, err)
		}
	}
	inW.Close()
	if err := <-done; err != nil {
		t.Fatal(err)
	}
}
