// Package errors is a minimal stand-in for github.com/juju/juju/internal/errors,
// which the copied Juju files import but which can't be imported from outside
// juju/juju. It keeps the parts the copied code uses (New, Errorf, Capture and
// Error.Add) with the same Error() strings and Is/As behaviour. Juju's version
// also records call-site locations for ErrorStack; that isn't reproduced
// because nothing in the runner prints it.
package errors

import (
	stderrors "errors"
	"fmt"
)

// Error mirrors juju/internal/errors.Error.
type Error interface {
	error
	Add(err error) Error
	Unwrap() error
}

type link struct{ error }

func (l link) Add(err error) Error {
	if err == nil {
		return l
	}
	return link{annotated{l.error, err}}
}

func (l link) Unwrap() error { return l.error }

type annotated struct {
	error
	annotation error
}

func (a annotated) Unwrap() []error { return []error{a.error, a.annotation} }

// New mirrors juju/internal/errors.New.
func New(text string) Error { return link{stderrors.New(text)} }

// Errorf mirrors juju/internal/errors.Errorf (a proxy for fmt.Errorf).
func Errorf(format string, a ...any) Error { return link{fmt.Errorf(format, a...)} }

// Capture mirrors juju/internal/errors.Capture without the location record.
func Capture(err error) error {
	if err == nil {
		return nil
	}
	return err
}

// Is and As proxy the standard library.
func Is(err, target error) bool     { return stderrors.Is(err, target) }
func As(err error, target any) bool { return stderrors.As(err, target) }
