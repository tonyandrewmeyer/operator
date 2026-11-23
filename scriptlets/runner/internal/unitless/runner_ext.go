// Not copied from Juju: runner-only additions to the copied unitless package.
// They live in the same package so that the runner can drive Juju's own
// starformExecutor.Handle, IntentCollector and set_status code unchanged.

package unitless

import (
	"context"

	"github.com/canonical/starform/starform"

	domainunitless "scriptlet-runner/internal/domainunitless"
)

// Default limits, as the unitless manifold passes them to the worker.
const (
	DefaultMaxAllocs = defaultMaxAllocs
	DefaultMaxSteps  = defaultMaxSteps
)

// IntentSetStatusType re-exports the set-status intent type.
const IntentSetStatusType = IntentSetStatus

// ProbeExecutor is a Juju starformExecutor whose limits can be changed
// between Handle calls, so the runner can bisect for the steps and
// allocations an event uses without reloading the sources (which would run
// every init() again on a fresh budget, under the same probe limit).
//
// It is built exactly as NewStarformExecutor builds its script set (same app
// object, safety flags and no modules), except that the runner keeps the
// options pointer. starform reads MaxSteps and MaxAllocs from that pointer
// each time Handle makes a thread, so changing them takes effect on the next
// Handle. That relies on starform keeping the pointer it was given, which
// TestProbeLimitsTakeEffect checks.
type ProbeExecutor struct {
	exec    *starformExecutor
	options *starform.ScriptSetOptions
}

// NewProbeExecutor mirrors NewStarformExecutor.
func NewProbeExecutor(ctx context.Context, cfg ExecutorConfig) (*ProbeExecutor, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	options := &starform.ScriptSetOptions{
		App:            newAppObject(),
		Logger:         cfg.Logger,
		RequiredSafety: requiredSafety,
		MaxAllocs:      cfg.MaxAllocs,
		MaxSteps:       cfg.MaxSteps,
	}
	scriptSet, err := starform.NewScriptSet(options)
	if err != nil {
		return nil, err
	}
	sources := make([]starform.ScriptSource, len(cfg.Scriptlet.Sources))
	for i, s := range cfg.Scriptlet.Sources {
		sources[i] = s
	}
	if err := scriptSet.LoadSources(ctx, sources); err != nil {
		return nil, err
	}
	return &ProbeExecutor{exec: &starformExecutor{scriptSet: scriptSet}, options: options}, nil
}

// Handle runs Juju's starformExecutor.Handle with the given limits. The
// logger can't be swapped here: starform binds print and debug to the logger
// it had when the sources were loaded, so callers pass a logger they can
// redirect.
func (p *ProbeExecutor) Handle(ctx context.Context, event domainunitless.Event, maxSteps, maxAllocs int64) ([]Intent, error) {
	p.options.MaxSteps = maxSteps
	p.options.MaxAllocs = maxAllocs
	return p.exec.Handle(ctx, event)
}

// IsObserved reports whether any init() observed the event.
func (p *ProbeExecutor) IsObserved(name string) bool {
	return p.exec.scriptSet.IsObserved(name)
}
