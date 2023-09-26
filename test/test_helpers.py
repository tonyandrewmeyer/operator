# Copyright 2019-2021 Canonical Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pathlib
import shutil
import subprocess
import tempfile
import typing
import unittest

import ops
from ops.model import _ModelBackend
from ops.storage import SQLiteStorage


class FakeScriptTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):  # type: ignore
        super().__init__(*args, **kwargs)  # type: ignore
        fake_script_path = tempfile.mkdtemp('-fake_script')
        self._old_path = os.environ["PATH"]
        os.environ['PATH'] = os.pathsep.join([fake_script_path, os.environ["PATH"]])
        self.fake_script_path = pathlib.Path(fake_script_path)

    def setUp(self):
        def cleanup():
            shutil.rmtree(self.fake_script_path.as_posix())
            os.environ['PATH'] = self._old_path

        self.addCleanup(cleanup)

    def fake_script(self, name: str, content: str):
        template_args = {
            'name': name,
            'path': self.fake_script_path.as_posix(),
            'content': content,
        }

        path = self.fake_script_path / name
        with path.open('wt') as f:
            # Before executing the provided script, dump the provided arguments in calls.txt.
            # ASCII 1E is RS 'record separator', and 1C is FS 'file separator', which
            # seem appropriate.
            f.write('''#!/bin/sh
    {{ printf {name}; printf "\\036%s" "$@"; printf "\\034"; }} >> {path}/calls.txt
    {content}'''.format_map(template_args))
        os.chmod(str(path), 0o755)
        # TODO: this hardcodes the path to bash.exe, which works for now but might
        #       need to be set via environ or something like that.
        path.with_suffix(".bat").write_text(
            f'@"C:\\Program Files\\git\\bin\\bash.exe" {path} %*\n')

    def fake_script_calls(self, clear: bool = False) -> typing.List[typing.List[str]]:
        calls_file = self.fake_script_path / 'calls.txt'
        if not calls_file.exists():
            return []

        # newline and encoding forced to linuxy defaults because on
        # windows they're written from git-bash
        with calls_file.open('r+t', newline='\n', encoding='utf8') as f:
            calls = [line.split('\x1e') for line in f.read().split('\x1c')[:-1]]
            if clear:
                f.truncate(0)
        return calls


class FakeScriptTest(FakeScriptTestCase):

    def test_fake_script_works(self):
        self.fake_script('foo', 'echo foo runs')
        self.fake_script('bar', 'echo bar runs')
        # subprocess.getoutput goes via the shell, so it needs to be
        # something both sh and CMD understand
        output = subprocess.getoutput('foo a "b c " && bar "d e" f')
        self.assertEqual(output, 'foo runs\nbar runs')
        self.assertEqual(self.fake_script_calls(), [
            ['foo', 'a', 'b c '],
            ['bar', 'd e', 'f'],
        ])

    def test_fake_script_clear(self):
        self.fake_script('foo', 'echo foo runs')

        output = subprocess.getoutput('foo a "b c"')
        self.assertEqual(output, 'foo runs')

        self.assertEqual(self.fake_script_calls(clear=True), [['foo', 'a', 'b c']])

        self.fake_script('bar', 'echo bar runs')

        output = subprocess.getoutput('bar "d e" f')
        self.assertEqual(output, 'bar runs')

        self.assertEqual(self.fake_script_calls(clear=True), [['bar', 'd e', 'f']])

        self.assertEqual(self.fake_script_calls(clear=True), [])


class BaseTestCase(unittest.TestCase):

    def create_framework(self,
                         *,
                         model: typing.Optional[ops.Model] = None,
                         tmpdir: typing.Optional[pathlib.Path] = None):
        """Create a Framework object.

        By default operate in-memory; pass a temporary directory via the 'tmpdir'
        parameter if you wish to instantiate several frameworks sharing the
        same dir (e.g. for storing state).
        """
        if tmpdir is None:
            data_fpath = ":memory:"
            charm_dir = 'non-existant'
        else:
            data_fpath = tmpdir / "framework.data"
            charm_dir = tmpdir

        framework = ops.Framework(
            SQLiteStorage(data_fpath),
            charm_dir,
            meta=model._cache._meta if model else ops.CharmMeta(),
            model=model)  # type: ignore
        self.addCleanup(framework.close)
        return framework

    def create_model(self):
        """Create a Model object."""
        backend = _ModelBackend(unit_name='myapp/0')
        meta = ops.CharmMeta()
        model = ops.Model(meta, backend)
        return model
