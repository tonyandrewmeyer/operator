package unitless

import (
	"context"
	"errors"
	"testing"

	"github.com/canonical/starform/starform"
	"github.com/canonical/starlark/starlark"

	domainunitless "scriptlet-runner/internal/domainunitless"
)

type nopLogger struct{}

func (nopLogger) Log(context.Context, starform.LogEntry) {}

// ProbeExecutor relies on starform keeping the options pointer it was given
// and reading MaxSteps/MaxAllocs from it on every Handle.
func TestProbeLimitsTakeEffect(t *testing.T) {
	ctx := context.Background()
	p, err := NewProbeExecutor(ctx, ExecutorConfig{
		Scriptlet: domainunitless.ScriptletApplication{Sources: []domainunitless.ScriptSource{{LoadPath: "hook.star", Source: `
def h(event):
    x = [i for i in range(1000)]
    juju.set_status("active")
def init():
    juju.observe("config_changed", h)
`}}},
		MaxSteps: DefaultMaxSteps, MaxAllocs: DefaultMaxAllocs, Logger: nopLogger{},
	})
	if err != nil {
		t.Fatal(err)
	}
	ev := domainunitless.Event{Name: "config_changed"}
	if in, err := p.Handle(ctx, ev, DefaultMaxSteps, DefaultMaxAllocs); err != nil || len(in) != 1 {
		t.Fatalf("default limits: %v %v", in, err)
	}
	var se *starlark.StepsSafetyError
	if _, err := p.Handle(ctx, ev, 100, DefaultMaxAllocs); !errors.As(err, &se) {
		t.Fatalf("100 steps: %v", err)
	}
	var ae *starlark.AllocsSafetyError
	if _, err := p.Handle(ctx, ev, DefaultMaxSteps, 1000); !errors.As(err, &ae) {
		t.Fatalf("1000 bytes: %v", err)
	}
	if _, err := p.Handle(ctx, ev, DefaultMaxSteps, DefaultMaxAllocs); err != nil {
		t.Fatalf("back to default: %v", err)
	}
	if !p.IsObserved("config_changed") || p.IsObserved("update_status") {
		t.Fatal("IsObserved")
	}
}
