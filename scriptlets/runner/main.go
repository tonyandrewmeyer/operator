// Command runner runs a unitless Juju charm's Starlark scriptlets for one
// event, the way the Juju controller does, and reports the result as JSON.
//
//	runner run --charm DIR|FILE.charm --event FILE.json [flags]
//	runner load --charm DIR|FILE.charm [flags]
//	runner serve            # newline-delimited JSON on stdin/stdout
//	runner version
//
// See README.md for the protocol.
package main

import (
	"bufio"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"runtime"
	"runtime/debug"

	"scriptlet-runner/internal/run"
)

// JujuCommit is the juju/juju commit the files in internal/unitless and
// internal/domainunitless were copied from.
const JujuCommit = "145ae4877aba52e983dd426bf7a5009d7363fd6d"

// JujuTag is the juju/juju release tag that JujuCommit is, or "" when the
// commit isn't in any release yet (reported as null).
const JujuTag = ""

// JujuVersion is the version in Juju's core/version/version.go at
// JujuCommit: what a controller built from that commit reports.
const JujuVersion = "4.2-beta1"

// Versions are reported by "runner version" and the serve "version" op.
type Versions struct {
	Protocol   int    `json:"protocol"`
	JujuCommit string `json:"juju_commit"`
	// JujuTag is null when the commit isn't in a Juju release yet.
	JujuTag     *string `json:"juju_tag"`
	JujuVersion string  `json:"juju_version"`
	Starform    string  `json:"starform"`
	Starlark    string  `json:"starlark"`
	Go          string  `json:"go"`
}

func versions() Versions {
	v := Versions{Protocol: run.ProtocolVersion, JujuCommit: JujuCommit, JujuVersion: JujuVersion, Go: runtime.Version()}
	if JujuTag != "" {
		tag := JujuTag
		v.JujuTag = &tag
	}
	if bi, ok := debug.ReadBuildInfo(); ok {
		for _, d := range bi.Deps {
			switch d.Path {
			case "github.com/canonical/starform":
				v.Starform = d.Version
			case "github.com/canonical/starlark":
				v.Starlark = d.Version
			}
		}
	}
	return v
}

func main() {
	if len(os.Args) < 2 {
		usage()
	}
	switch os.Args[1] {
	case "run":
		os.Exit(cmdRun(os.Args[2:]))
	case "load":
		os.Exit(cmdLoad(os.Args[2:]))
	case "serve":
		if err := serve(os.Stdin, os.Stdout); err != nil {
			fmt.Fprintln(os.Stderr, "runner serve:", err)
			os.Exit(1)
		}
	case "version":
		writeJSON(os.Stdout, versions(), true)
	default:
		usage()
	}
}

func usage() {
	fmt.Fprintln(os.Stderr, "usage: runner run --charm PATH --event FILE.json [--max-steps N] [--max-allocs N] [--no-measure] [--cap-factor N] [--skip-charm-checks] [--pretty]\n       runner load --charm PATH [--max-steps N] [--max-allocs N] [--no-measure] [--cap-factor N] [--skip-charm-checks] [--pretty]\n       runner serve\n       runner version")
	os.Exit(2)
}

// commonFlags are the options shared by the run and load subcommands.
type commonFlags struct {
	charm                         *string
	maxSteps, maxAllocs, capFac   *int64
	noMeasure, skipChecks, pretty *bool
}

func addCommonFlags(fs *flag.FlagSet) commonFlags {
	return commonFlags{
		charm:      fs.String("charm", "", "charm directory or .charm file"),
		maxSteps:   fs.Int64("max-steps", 0, "step limit (default Juju's 100000)"),
		maxAllocs:  fs.Int64("max-allocs", 0, "allocation limit in bytes (default Juju's 10 MiB)"),
		noMeasure:  fs.Bool("no-measure", false, "don't bisect for steps and allocations used"),
		capFac:     fs.Int64("cap-factor", 0, "when a limit is hit, search up to this multiple of it (default 10)"),
		skipChecks: fs.Bool("skip-charm-checks", false, "skip Juju's add-charm checks (assumes: [unitless], no hooks)"),
		pretty:     fs.Bool("pretty", false, "indent the JSON output"),
	}
}

func (c commonFlags) options() run.Options {
	measure := !*c.noMeasure
	return run.Options{
		MaxSteps: *c.maxSteps, MaxAllocs: *c.maxAllocs, Measure: &measure,
		CapFactor: *c.capFac, SkipCharmChecks: *c.skipChecks,
	}
}

func cmdRun(args []string) int {
	fs := flag.NewFlagSet("run", flag.ContinueOnError)
	c := addCommonFlags(fs)
	eventFile := fs.String("event", "", `event JSON file ({"name": ..., "attrs": {...}}), or - for stdin`)
	if err := fs.Parse(args); err != nil || *c.charm == "" || *eventFile == "" {
		usage()
	}
	var raw []byte
	var err error
	if *eventFile == "-" {
		raw, err = io.ReadAll(os.Stdin)
	} else {
		raw, err = os.ReadFile(*eventFile)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "runner:", err)
		return 2
	}
	var ev run.Event
	if err := json.Unmarshal(raw, &ev); err != nil {
		fmt.Fprintln(os.Stderr, "runner: event file:", err)
		return 2
	}
	res := run.New().Run(context.Background(), *c.charm, ev, c.options())
	writeJSON(os.Stdout, res, *c.pretty)
	return 0
}

func cmdLoad(args []string) int {
	fs := flag.NewFlagSet("load", flag.ContinueOnError)
	c := addCommonFlags(fs)
	if err := fs.Parse(args); err != nil || *c.charm == "" || fs.NArg() != 0 {
		usage()
	}
	writeJSON(os.Stdout, run.New().Load(context.Background(), *c.charm, c.options()), *c.pretty)
	return 0
}

// Request is one line of serve input.
type Request struct {
	ID      json.RawMessage `json:"id,omitempty"`
	Op      string          `json:"op"` // "run" (default), "load", "version", "shutdown"
	Charm   string          `json:"charm"`
	Event   run.Event       `json:"event"`
	Options run.Options     `json:"options"`
}

type errorResponse struct {
	Protocol int             `json:"protocol"`
	ID       json.RawMessage `json:"id,omitempty"`
	OK       bool            `json:"ok"`
	Error    *run.Error      `json:"error"`
}

type versionResponse struct {
	ID json.RawMessage `json:"id,omitempty"`
	OK bool            `json:"ok"`
	Versions
}

func serve(in io.Reader, out io.Writer) error {
	r := run.New()
	sc := bufio.NewScanner(in)
	sc.Buffer(make([]byte, 0, 1<<20), 64<<20)
	w := bufio.NewWriter(out)
	for sc.Scan() {
		line := sc.Bytes()
		if len(line) == 0 {
			continue
		}
		var req Request
		if err := json.Unmarshal(line, &req); err != nil {
			writeJSON(w, errorResponse{Protocol: run.ProtocolVersion, Error: &run.Error{Kind: "invalid_input", Phase: "input", Message: "request: " + err.Error()}}, false)
		} else {
			switch req.Op {
			case "", "run":
				res := r.Run(context.Background(), req.Charm, req.Event, req.Options)
				res.ID = req.ID
				writeJSON(w, res, false)
			case "load":
				res := r.Load(context.Background(), req.Charm, req.Options)
				res.ID = req.ID
				writeJSON(w, res, false)
			case "version":
				writeJSON(w, versionResponse{ID: req.ID, OK: true, Versions: versions()}, false)
			case "shutdown":
				writeJSON(w, versionResponse{ID: req.ID, OK: true, Versions: versions()}, false)
				return w.Flush()
			default:
				writeJSON(w, errorResponse{Protocol: run.ProtocolVersion, ID: req.ID, Error: &run.Error{Kind: "invalid_input", Phase: "input", Message: fmt.Sprintf("unknown op %q", req.Op)}}, false)
			}
		}
		if err := w.Flush(); err != nil {
			return err
		}
	}
	return sc.Err()
}

func writeJSON(w io.Writer, v any, pretty bool) {
	enc := json.NewEncoder(w)
	enc.SetEscapeHTML(false)
	if pretty {
		enc.SetIndent("", "  ")
	}
	_ = enc.Encode(v)
}
