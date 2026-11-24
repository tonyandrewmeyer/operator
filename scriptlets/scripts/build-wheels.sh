#!/bin/bash
# Copyright 2026 Canonical Ltd.
# See LICENSE file for licensing details.

# Build the ops-testing-scriptlets sdist, and a wheel for each platform, into
# the directory given as the only argument. The runner is cross-compiled, so
# every wheel can be built on one machine. Needs uv and Go.

set -xueo pipefail

out=${1:?usage: build-wheels.sh OUT_DIR}

# With no GOOS or GOARCH this is the host's wheel, and it's built from the
# sdist, so it checks that the sdist is complete.
env -u GOOS -u GOARCH uv build --package ops-testing-scriptlets --out-dir "$out"

host=$(go env GOHOSTOS)/$(go env GOHOSTARCH)
for target in linux/amd64 linux/arm64 darwin/arm64; do
  if [ "$target" = "$host" ]; then
    continue
  fi
  GOOS=${target%/*} GOARCH=${target#*/} \
    uv build --package ops-testing-scriptlets --wheel --out-dir "$out"
done
