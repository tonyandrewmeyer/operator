// Package corelife is a minimal stand-in for github.com/juju/juju/core/life,
// providing only the Value type the copied Juju files use. The runner never
// sets it.
package corelife

// Value holds the lifecycle state of an entity.
type Value string
