// Package coreerrors is a minimal stand-in for github.com/juju/juju/core/errors,
// providing only the NotValid sentinel the copied Juju files use. Juju's
// NotValid is juju/errors.NotValid, a ConstError with the text "not valid";
// it's only ever added to an error chain with Add, so its text doesn't appear
// in messages.
package coreerrors

// ConstError is a sentinel error type.
type ConstError string

func (e ConstError) Error() string { return string(e) }

// NotValid represents an error when something is not valid.
const NotValid = ConstError("not valid")
