#!/bin/sh
# Check that the files copied from juju/juju still match Juju's source.
#
#   scripts/check-juju-copy.sh /path/to/juju [REF]
#
# REF defaults to the commit recorded in main.go (JujuCommit). Pass a tag or
# branch to see what changed upstream since the copy. Exits non-zero on any
# difference other than the rewritten import paths and the header.
set -eu
juju=$1
here=$(cd "$(dirname "$0")/.." && pwd)
ref=${2:-$(sed -n 's/^const JujuCommit = "\(.*\)"$/\1/p' "$here/main.go")}
status=0
check() { # juju-path local-path
    tmp=$(mktemp)
    git -C "$juju" show "$ref:$1" | sed \
        -e 's#"github.com/juju/juju/internal/errors"#"scriptlet-runner/internal/errors"#' \
        -e 's#"github.com/juju/juju/core/errors"#"scriptlet-runner/internal/coreerrors"#' \
        -e 's#"github.com/juju/juju/domain/unitless"#"scriptlet-runner/internal/domainunitless"#' \
        -e 's#"github.com/juju/juju/core/application"#"scriptlet-runner/internal/coreapplication"#' \
        -e 's#"github.com/juju/juju/core/life"#"scriptlet-runner/internal/corelife"#' >"$tmp"
    if tail -n +5 "$here/$2" | diff -u "$tmp" - >/dev/null; then
        echo "same: $1"
    else
        echo "DIFFERS: $1"
        tail -n +5 "$here/$2" | diff -u "$tmp" - || true
        status=1
    fi
    rm -f "$tmp"
}
for f in executor intent setstatus scriptlet; do
    check "internal/worker/unitless/$f.go" "internal/unitless/$f.go"
done
check domain/unitless/types.go internal/domainunitless/types.go
# Juju's pins must match go.mod.
for mod in github.com/canonical/starform github.com/canonical/starlark; do
    theirs=$(git -C "$juju" show "$ref:go.mod" | awk -v m="$mod" '$1 == m {print $2}')
    ours=$(awk -v m="$mod" '$1 == m {print $2}' "$here/go.mod")
    if [ "$theirs" = "$ours" ]; then echo "same: $mod $ours"; else echo "DIFFERS: $mod juju=$theirs runner=$ours"; status=1; fi
done
# JujuVersion and JujuTag must describe the recorded commit (only checked
# when REF is the recorded commit; another REF is a look-ahead).
commit=$(sed -n 's/^const JujuCommit = "\(.*\)"$/\1/p' "$here/main.go")
if [ "$(git -C "$juju" rev-parse "$ref^{commit}")" = "$commit" ]; then
    theirs=$(git -C "$juju" show "$ref:core/version/version.go" | sed -n 's/^const version = "\(.*\)"$/\1/p')
    ours=$(sed -n 's/^const JujuVersion = "\(.*\)"$/\1/p' "$here/main.go")
    if [ "$theirs" = "$ours" ]; then echo "same: version $ours"; else echo "DIFFERS: version juju=$theirs runner=$ours"; status=1; fi
    tag=$(sed -n 's/^const JujuTag = "\(.*\)"$/\1/p' "$here/main.go")
    tags=$(git -C "$juju" tag --points-at "$commit" 'v*' | tr '\n' ' ')
    if [ -z "$tag" ] && [ -z "$tags" ]; then echo "same: no tag at $commit"
    elif [ -n "$tag" ] && git -C "$juju" tag --points-at "$commit" | grep -qx "$tag"; then echo "same: tag $tag"
    else echo "DIFFERS: tag runner=${tag:-none} juju=${tags:-none}"; status=1; fi
fi
exit $status
