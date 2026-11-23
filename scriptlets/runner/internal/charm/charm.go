// Package charm reads a unitless charm's scriptlet sources from a charm
// directory or a packed .charm archive, using the rules Juju applies when it
// reads a charm archive and validates it on add-charm.
//
// Juju references (juju/juju main at 1eb8e4039dfa):
//   - domain/deployment/charm/charmarchive.go, readScriptletSources: every
//     archive member under "scriptlets/" whose path.Ext is ".star" and that is
//     not a directory or a symlink, sorted by member name, with the
//     "scriptlets/" prefix stripped to give the load path.
//   - domain/application/service/charm.go, validateScriptletSources: no
//     scriptlets alongside hooks/ or dispatch; scriptlets iff assumes
//     includes "unitless" (searching any-of/all-of groups, as
//     core/assumes.HasFeature does).
//
// starform then skips any source whose path doesn't end in ".star" and
// rejects invalid load paths when the sources are loaded; that happens in
// the runner, not here, so that the error is starform's own.
package charm

import (
	"archive/zip"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	"gopkg.in/yaml.v3"
)

const scriptletsPrefix = "scriptlets/"

// Source is one scriptlet file: its load path (relative to scriptlets/,
// with "/" separators) and its content.
type Source struct {
	Path    string
	Content []byte
}

// Charm is what the runner needs from a charm.
type Charm struct {
	Root    string
	Sources []Source
	// MetadataFile is the file assumes was read from, or "" if none.
	MetadataFile string
}

// InvalidError is a charm that Juju would refuse at add-charm time.
type InvalidError struct{ Msg string }

func (e *InvalidError) Error() string { return e.Msg }

// Load reads a charm directory or a .charm (zip) file. If check is true, it
// applies Juju's add-charm scriptlet rules and returns an *InvalidError if
// they fail.
func Load(root string, check bool) (*Charm, error) {
	info, err := os.Stat(root)
	if err != nil {
		return nil, err
	}
	var (
		ch           *Charm
		hasHooks     bool
		meta         []byte
		metadataFile string
	)
	if info.IsDir() {
		ch, err = loadDir(root)
		if err != nil {
			return nil, err
		}
		hasHooks = exists(filepath.Join(root, "hooks"), true) || exists(filepath.Join(root, "dispatch"), false)
		// A packed charm always has metadata.yaml (charmcraft generates it
		// from charmcraft.yaml); a source tree may only have charmcraft.yaml.
		for _, name := range []string{"metadata.yaml", "charmcraft.yaml"} {
			if b, err := os.ReadFile(filepath.Join(root, name)); err == nil {
				meta, metadataFile = b, name
				break
			}
		}
	} else {
		ch, hasHooks, meta, err = loadArchive(root)
		if err != nil {
			return nil, err
		}
		if meta != nil {
			metadataFile = "metadata.yaml"
		}
	}
	ch.MetadataFile = metadataFile
	if !check {
		return ch, nil
	}

	hasScriptlets := len(ch.Sources) > 0
	if hasScriptlets && hasHooks {
		return nil, &InvalidError{"charm has scriptlets alongside hooks or a dispatch file"}
	}
	if meta == nil {
		return nil, &InvalidError{"charm has no metadata.yaml or charmcraft.yaml, so assumes can't be checked (use --skip-charm-checks to run anyway)"}
	}
	unitless, err := assumesFeature(meta, "unitless")
	if err != nil {
		return nil, &InvalidError{fmt.Sprintf("reading %s: %v", metadataFile, err)}
	}
	if hasScriptlets && !unitless {
		return nil, &InvalidError{"charm has scriptlets but does not assume unitless"}
	}
	if unitless && !hasScriptlets {
		return nil, &InvalidError{"charm assumes unitless but has no scriptlets"}
	}
	return ch, nil
}

func exists(p string, dir bool) bool {
	fi, err := os.Lstat(p)
	if err != nil {
		return false
	}
	if dir {
		return fi.IsDir()
	}
	return !fi.IsDir()
}

// loadDir applies readScriptletSources' selection to a directory tree:
// regular files only (symlinks, to files or directories, are skipped and not
// followed), path.Ext == ".star", sorted as the archive member names would be.
func loadDir(root string) (*Charm, error) {
	ch := &Charm{Root: root}
	base := filepath.Join(root, "scriptlets")
	if _, err := os.Lstat(base); errors.Is(err, fs.ErrNotExist) {
		return ch, nil
	}
	var names []string
	err := filepath.WalkDir(base, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if !d.Type().IsRegular() {
			return nil // directories are walked; symlinks and other types skipped
		}
		rel, err := filepath.Rel(root, p)
		if err != nil {
			return err
		}
		name := filepath.ToSlash(rel)
		if strings.HasPrefix(name, scriptletsPrefix) && path.Ext(name) == ".star" {
			names = append(names, name)
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	sort.Strings(names)
	for _, name := range names {
		b, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(name)))
		if err != nil {
			return nil, fmt.Errorf("reading scriptlet source %q: %w", name, err)
		}
		ch.Sources = append(ch.Sources, Source{Path: strings.TrimPrefix(name, scriptletsPrefix), Content: b})
	}
	return ch, nil
}

func loadArchive(file string) (*Charm, bool, []byte, error) {
	zr, err := zip.OpenReader(file)
	if err != nil {
		return nil, false, nil, err
	}
	defer zr.Close()
	ch := &Charm{Root: file}
	var files []*zip.File
	var meta []byte
	hasHooks := false
	for _, f := range zr.File {
		mode := f.FileInfo().Mode()
		if strings.HasPrefix(f.Name, scriptletsPrefix) && path.Ext(f.Name) == ".star" &&
			!mode.IsDir() && mode&fs.ModeSymlink == 0 {
			files = append(files, f)
		}
		// archiveHasDirectory(zipr, "hooks") || archiveHasFile(zipr, "dispatch")
		if (f.Name == "dispatch" && !mode.IsDir()) || (f.Name == "hooks" && mode.IsDir()) || strings.HasPrefix(f.Name, "hooks/") {
			hasHooks = true
		}
		if f.Name == "metadata.yaml" {
			if meta, err = readZip(f); err != nil {
				return nil, false, nil, err
			}
		}
	}
	sort.Slice(files, func(i, j int) bool { return files[i].Name < files[j].Name })
	for _, f := range files {
		b, err := readZip(f)
		if err != nil {
			return nil, false, nil, fmt.Errorf("reading scriptlet source %q: %w", f.Name, err)
		}
		ch.Sources = append(ch.Sources, Source{Path: strings.TrimPrefix(f.Name, scriptletsPrefix), Content: b})
	}
	return ch, hasHooks, meta, nil
}

func readZip(f *zip.File) ([]byte, error) {
	r, err := f.Open()
	if err != nil {
		return nil, err
	}
	defer r.Close()
	return io.ReadAll(r)
}

// featureName matches the start of an assumes feature expression, such as
// "unitless" or "juju >= 4.2".
var featureName = regexp.MustCompile(`^\s*([a-z][a-z0-9-]*)`)

func assumesFeature(meta []byte, feature string) (bool, error) {
	var doc struct {
		Assumes []any `yaml:"assumes"`
	}
	if err := yaml.Unmarshal(meta, &doc); err != nil {
		return false, err
	}
	return exprHasFeature(doc.Assumes, feature), nil
}

func exprHasFeature(v any, feature string) bool {
	switch v := v.(type) {
	case string:
		m := featureName.FindStringSubmatch(v)
		return m != nil && m[1] == feature
	case []any:
		for _, x := range v {
			if exprHasFeature(x, feature) {
				return true
			}
		}
	case map[string]any:
		for _, key := range []string{"any-of", "all-of"} {
			if exprHasFeature(v[key], feature) {
				return true
			}
		}
	}
	return false
}
