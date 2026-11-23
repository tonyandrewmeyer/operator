// Package coreapplication is a minimal stand-in for
// github.com/juju/juju/core/application, providing only the UUID type the
// copied Juju files use. The runner never sets it.
package coreapplication

// UUID represents an application unique identifier.
type UUID string
