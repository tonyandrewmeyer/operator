package charm

import (
	"archive/zip"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func write(t *testing.T, root string, files map[string]string) {
	t.Helper()
	for name, content := range files {
		p := filepath.Join(root, filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
}

func paths(ch *Charm) []string {
	out := []string{}
	for _, s := range ch.Sources {
		out = append(out, s.Path)
	}
	return out
}

const meta = "assumes:\n  - unitless\n"

// The selection rules of readScriptletSources: recursive under scriptlets/,
// path.Ext == ".star" exactly, regular files only (no symlinks), sorted by
// the full name, "scriptlets/" stripped. Load-path validity is starform's
// job and isn't checked here.
func TestSelectionRules(t *testing.T) {
	root := t.TempDir()
	write(t, root, map[string]string{
		"charmcraft.yaml":            meta,
		"scriptlets/zeta.star":       "z = 1\n",
		"scriptlets/alpha.star":      "a = 1\n",
		"scriptlets/lib/b_mod.star":  "b = 1\n",
		"scriptlets/lib_x.star":      "x = 1\n",
		"scriptlets/notes.txt":       "not a scriptlet",
		"scriptlets/upper.STAR":      "u = 1\n",
		"scriptlets/archive.star.gz": "x",
		"scriptlets/bad-name.star":   "y = 1\n", // selected; starform rejects it at load
		"scriptlets/.hidden.star":    "h = 1\n", // selected (path.Ext is .star)
		"top.star":                   "outside scriptlets/",
		"lib/other.star":             "outside scriptlets/",
	})
	if err := os.Symlink("alpha.star", filepath.Join(root, "scriptlets/link.star")); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(filepath.Join(root, "scriptlets/lib"), filepath.Join(root, "scriptlets/linkdir")); err != nil {
		t.Fatal(err)
	}
	ch, err := Load(root, true)
	if err != nil {
		t.Fatal(err)
	}
	// Byte order of "scriptlets/<path>": "." < "a" < "b" < "l" ("lib/" < "lib_",
	// since "/" < "_") < "z".
	want := []string{".hidden.star", "alpha.star", "bad-name.star", "lib/b_mod.star", "lib_x.star", "zeta.star"}
	if got := paths(ch); !reflect.DeepEqual(got, want) {
		t.Fatalf("sources = %q, want %q", got, want)
	}
	if string(ch.Sources[1].Content) != "a = 1\n" {
		t.Fatalf("content = %q", ch.Sources[1].Content)
	}
}

func TestSelectionFromArchiveMatchesDirectory(t *testing.T) {
	root := t.TempDir()
	files := map[string]string{
		"metadata.yaml":           "name: x\n" + meta,
		"scriptlets/b.star":       "b = 1\n",
		"scriptlets/a/inner.star": "i = 1\n",
		"scriptlets/a.star":       "a = 1\n",
		"scriptlets/readme.md":    "no",
	}
	write(t, root, files)
	zpath := filepath.Join(t.TempDir(), "x.charm")
	f, err := os.Create(zpath)
	if err != nil {
		t.Fatal(err)
	}
	zw := zip.NewWriter(f)
	for name, content := range files {
		w, _ := zw.Create(name)
		w.Write([]byte(content))
	}
	// A symlink member must be skipped, as Juju does.
	hdr := &zip.FileHeader{Name: "scriptlets/link.star"}
	hdr.SetMode(os.ModeSymlink | 0o777)
	w, _ := zw.CreateHeader(hdr)
	w.Write([]byte("a.star"))
	zw.Close()
	f.Close()

	fromDir, err := Load(root, true)
	if err != nil {
		t.Fatal(err)
	}
	fromZip, err := Load(zpath, true)
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"a.star", "a/inner.star", "b.star"} // "." < "/"
	if got := paths(fromDir); !reflect.DeepEqual(got, want) {
		t.Fatalf("dir sources = %q, want %q", got, want)
	}
	if got := paths(fromZip); !reflect.DeepEqual(got, want) {
		t.Fatalf("zip sources = %q, want %q", got, want)
	}
}

func TestCharmChecks(t *testing.T) {
	for _, tc := range []struct {
		name  string
		files map[string]string
		err   string
	}{
		{"ok", map[string]string{"charmcraft.yaml": meta, "scriptlets/a.star": "a = 1\n"}, ""},
		{"metadata wins", map[string]string{"metadata.yaml": meta, "charmcraft.yaml": "assumes: []\n", "scriptlets/a.star": "x = 1\n"}, ""},
		{"any-of", map[string]string{"charmcraft.yaml": "assumes:\n  - any-of:\n    - unitless\n    - k8s-api\n", "scriptlets/a.star": "x = 1\n"}, ""},
		{"versioned feature isn't unitless", map[string]string{"charmcraft.yaml": "assumes:\n  - juju >= 4.2\n", "scriptlets/a.star": "x = 1\n"}, "does not assume unitless"},
		{"hooks", map[string]string{"charmcraft.yaml": meta, "scriptlets/a.star": "x = 1\n", "hooks/install": "#!/bin/sh\n"}, "alongside hooks"},
		{"dispatch", map[string]string{"charmcraft.yaml": meta, "scriptlets/a.star": "x = 1\n", "dispatch": "#!/bin/sh\n"}, "alongside hooks"},
		{"no scriptlets", map[string]string{"charmcraft.yaml": meta, "scriptlets/a.txt": "x"}, "has no scriptlets"},
		{"no metadata", map[string]string{"scriptlets/a.star": "x = 1\n"}, "no metadata.yaml"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			root := t.TempDir()
			write(t, root, tc.files)
			_, err := Load(root, true)
			if tc.err == "" {
				if err != nil {
					t.Fatal(err)
				}
				return
			}
			var inv *InvalidError
			if err == nil || !strings.Contains(err.Error(), tc.err) {
				t.Fatalf("err = %v, want %q", err, tc.err)
			}
			if !asInvalid(err, &inv) {
				t.Fatalf("err %T is not *InvalidError", err)
			}
			if _, err := Load(root, false); err != nil {
				t.Fatalf("with checks skipped: %v", err)
			}
		})
	}
}

func asInvalid(err error, target **InvalidError) bool {
	e, ok := err.(*InvalidError)
	*target = e
	return ok
}
