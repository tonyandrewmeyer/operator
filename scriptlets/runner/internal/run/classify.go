package run

import (
	"errors"
	"fmt"
	"regexp"
	"runtime/debug"
	"strings"

	"github.com/canonical/starlark/resolve"
	"github.com/canonical/starlark/starlark"
	"github.com/canonical/starlark/syntax"

	"scriptlet-runner/internal/coreerrors"
)

func isSteps(err error) bool {
	var e *starlark.StepsSafetyError
	return err != nil && errors.As(err, &e)
}

func isAllocs(err error) bool {
	var e *starlark.AllocsSafetyError
	return err != nil && errors.As(err, &e)
}

// isValidateErr reports an ExecutorConfig.Validate failure (for example an
// empty source file), which Juju tags with coreerrors.NotValid.
func isValidateErr(err error) bool {
	return err != nil && errors.Is(err, coreerrors.NotValid)
}

// Messages starform's sanitiseLoadPath and Load return. They're unexported,
// so match on text.
var loadPathMessages = []string{
	"path invalid, see https://github.com/canonical/starlark/blob/main/doc/valid-load-paths.md",
	`path contains "-", use "_" instead`,
	`path contains "\", use "/" instead`,
	"path contains redundant components",
	": file not found",
}

// starlark.UnpackArgs / UnpackPositionalArgs error shapes, after "<fn>: ".
var unpackArgsMessage = regexp.MustCompile(`^(missing argument for |got \d+ arguments, want |for parameter |got multiple values for keyword argument |unexpected keyword arguments?)`)

var cachedName = regexp.MustCompile(`cached-[0-9a-f]{8}\.star`)

func mapNames(s string, names map[string]string) string {
	return cachedName.ReplaceAllStringFunc(s, func(m string) string {
		if p, ok := names[m]; ok {
			return p
		}
		return m
	})
}

func classify(err error, phase string, names map[string]string) *Error {
	out := &Error{Phase: phase, JujuLog: err.Error()}
	pos := func(p syntax.Position) *Position {
		if !p.IsValid() {
			return nil
		}
		return &Position{Path: mapNames(p.Filename(), names), Line: p.Line, Col: p.Col}
	}

	var (
		evalErr   *starlark.EvalError
		synErr    syntax.Error
		synErrPtr *syntax.Error
		resErr    resolve.ErrorList
	)
	var panicErr *PanicError
	switch {
	case errors.As(err, &panicErr):
		out.Kind = "interpreter_panic"
		out.Message = mapNames(panicErr.Error(), names)
		out.Backtrace = panicErr.Stack
		out.JujuLog = ""
		return out
	case isValidateErr(err):
		out.Kind = "charm_invalid"
		out.Message = err.Error()
		out.JujuLog = ""
		return out
	case errors.As(err, &synErrPtr):
		out.Kind = "syntax_error"
		out.Position = pos(synErrPtr.Pos)
		out.Message = mapNames(synErrPtr.Error(), names)
		return out
	case errors.As(err, &synErr):
		out.Kind = "syntax_error"
		out.Position = pos(synErr.Pos)
		out.Message = mapNames(synErr.Error(), names)
		return out
	case errors.As(err, &resErr):
		out.Kind = "syntax_error" // resolver: undefined names, dialect violations
		out.Position = pos(resErr[0].Pos)
		out.Message = mapNames(resErr.Error(), names)
		return out
	case errors.As(err, &evalErr):
		out.Message = mapNames(evalErr.Msg, names)
		out.Backtrace = mapNames(evalErr.Backtrace(), names)
		stack := evalErr.CallStack
		// Innermost frame in a script, and the builtin on top if there is one.
		for i := len(stack) - 1; i >= 0; i-- {
			if stack[i].Pos.Filename() == "<builtin>" {
				if i == len(stack)-1 {
					out.Builtin = stack[i].Name
				}
				continue
			}
			out.Position = pos(stack[i].Pos)
			break
		}
	default:
		out.Message = mapNames(err.Error(), names)
	}

	switch {
	case isSteps(err):
		out.Kind = "step_limit"
	case isAllocs(err):
		out.Kind = "alloc_limit"
	case strings.HasPrefix(err.Error(), "handling event ") && strings.Contains(err.Error(), "cannot handle "):
		out.Kind = "invalid_event"
	case containsAny(err.Error(), loadPathMessages):
		out.Kind = "load_path_error"
	case out.Builtin != "" && unpackArgsMessage.MatchString(strings.TrimPrefix(out.Message, out.Builtin+": ")):
		out.Kind = "builtin_argument_error"
	case evalErr != nil:
		out.Kind = "runtime_error"
	default:
		out.Kind = "internal"
	}
	return out
}

func containsAny(s string, subs []string) bool {
	for _, sub := range subs {
		if strings.Contains(s, sub) {
			return true
		}
	}
	return false
}

// PanicError is a panic inside starform or Starlark, recovered by the
// runner. Juju doesn't recover these: in the controller the panic would
// unwind the unitless worker's goroutine and take the process down.
type PanicError struct {
	Value any
	Stack string
}

func (e *PanicError) Error() string { return fmt.Sprintf("panic: %v", e.Value) }

func isPanic(err error) bool {
	var p *PanicError
	return err != nil && errors.As(err, &p)
}

// guard runs f, turning a panic into a *PanicError.
func guard(f func() error) (err error) {
	defer func() {
		if v := recover(); v != nil {
			err = &PanicError{Value: v, Stack: string(debug.Stack())}
		}
	}()
	return f()
}

// isAllocsOrPanic is the allocation probe's failure test. The only panic
// seen so far (starlark 8c33ed2, Call growing thread.stack) happens when
// AddAllocs fails, so a panicking probe failed on allocations.
func isAllocsOrPanic(err error) bool { return isAllocs(err) || isPanic(err) }
