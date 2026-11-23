// Code copied from juju/juju internal/worker/unitless/scriptlet.go at commit 145ae4877aba52e983dd426bf7a5009d7363fd6d.
// Do not edit: only the import paths of Juju-internal packages are rewritten
// (see scripts/check-juju-copy.sh). Everything after this header is Juju's.

// Copyright 2026 Canonical Ltd.
// Licensed under the AGPLv3, see LICENCE file for details.

package unitless

const (
	defaultMaxAllocs = 10 * 1024 * 1024
	defaultMaxSteps  = 100_000
)
