# Copyright 2026 Canonical Ltd.
# See LICENSE file for licensing details.

"""The pytest plugin, run through pytester."""

from __future__ import annotations

import pytest

from .conftest import TESTDATA, skip_if_bundled

STEP_LIMIT = TESTDATA / 'step_limit'
SET_STATUS = TESTDATA / 'set_status'

# 5,545 of 100,000 steps: 5.5% of the limit.
RELATION_100 = """
import json, pathlib

def relation_changed_100():
    data = json.loads(pathlib.Path({events!r}).read_text())
    return data['name'], data['attrs']
""".format(events=str(TESTDATA / 'events' / 'relation_changed_100.json'))


def test_fixture_from_ini(pytester: pytest.Pytester):
    pytester.makeini(f'[pytest]\nscriptlet_charm = {SET_STATUS}\n')
    pytester.makepyfile(
        """
        from scriptlet_testing import Context, Intent

        def test_it(scriptlet_ctx):
            assert isinstance(scriptlet_ctx, Context)
            out = scriptlet_ctx.run_raw('update_status')
            assert out.intents == [Intent('set-status', {'status': 'active', 'message': ''})]
            assert not out.measured
        """
    )
    result = pytester.runpytest()
    result.assert_outcomes(passed=1)
    # Not measuring, so no budget report.
    result.stdout.no_fnmatch_line('*scriptlet budget*')


def test_ini_path_is_relative_to_ini_file(pytester: pytest.Pytester):
    charm = pytester.path / 'charm'
    charm.mkdir()
    (charm / 'charmcraft.yaml').write_text('assumes: [unitless]\n')
    (charm / 'scriptlets').mkdir()
    (charm / 'scriptlets' / 'hook.star').write_text(
        'def h(event):\n    juju.set_status("waiting")\n\n'
        'def init():\n    juju.observe("start_up", h)\n'
    )
    pytester.makeini('[pytest]\nscriptlet_charm = charm\n')
    sub = pytester.mkdir('tests')
    (sub / 'test_rel.py').write_text(
        'def test_it(scriptlet_ctx):\n'
        '    assert scriptlet_ctx.charm_path.name == "charm"\n'
        '    assert scriptlet_ctx.run_raw("start_up").intents[0].args["status"] == "waiting"\n'
    )
    pytester.runpytest('tests').assert_outcomes(passed=1)


def test_marker_overrides_ini(pytester: pytest.Pytester):
    pytester.makeini(f'[pytest]\nscriptlet_charm = {SET_STATUS}\n')
    pytester.makepyfile(
        f"""
        import pytest

        @pytest.mark.scriptlet_charm({str(STEP_LIMIT)!r})
        def test_marked(scriptlet_ctx):
            assert scriptlet_ctx.charm_path.name == 'step_limit'

        def test_unmarked(scriptlet_ctx):
            assert scriptlet_ctx.charm_path.name == 'set_status'
        """
    )
    pytester.runpytest('--strict-markers').assert_outcomes(passed=2)


def test_rootdir_default(pytester: pytest.Pytester):
    pytester.makepyfile(
        """
        def test_it(scriptlet_ctx):
            pass
        """
    )
    # The rootdir has no scriptlets/, so loading fails (with no charm checks
    # passing either), as an error in the fixture.
    result = pytester.runpytest()
    result.assert_outcomes(errors=1)
    result.stdout.fnmatch_lines(['*InvalidCharmError*'])


def test_fixture_closes_runner(pytester: pytest.Pytester):
    pytester.makeini(f'[pytest]\nscriptlet_charm = {SET_STATUS}\n')
    pytester.makepyfile(
        """
        seen = []

        def test_a(scriptlet_ctx):
            seen.append(scriptlet_ctx)

        def test_b(scriptlet_ctx):
            assert seen[0].closed
            assert scriptlet_ctx is not seen[0]
        """
    )
    pytester.runpytest('-p', 'no:randomly').assert_outcomes(passed=2)


def test_header_has_versions(pytester: pytest.Pytester):
    pytester.makepyfile('def test_it(): pass')
    result = pytester.runpytest()
    result.stdout.fnmatch_lines([
        'scriptlet runner: *runner',
        'scriptlet runner versions: juju 4.2-beta1 (145ae4877aba, untagged), '
        'starform v0.0.0-*, '
        'starlark v0.0.0-*, go1.*, protocol 1',
    ])


@skip_if_bundled
def test_header_without_runner(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv('SCRIPTLET_TESTING_RUNNER')
    pytester.makepyfile('def test_it(): pass')
    result = pytester.runpytest()
    result.stdout.fnmatch_lines(['scriptlet runner: none (no scriptlet runner*'])
    result.assert_outcomes(passed=1)


def test_budget_off_by_default(pytester: pytest.Pytester):
    pytester.makeini(f'[pytest]\nscriptlet_charm = {STEP_LIMIT}\n')
    pytester.makepyfile(
        RELATION_100
        + """
def test_it(scriptlet_ctx):
    out = scriptlet_ctx.run_raw(*relation_changed_100())
    assert not out.measured
"""
    )
    result = pytester.runpytest()
    result.assert_outcomes(passed=1)
    result.stdout.no_fnmatch_line('*scriptlet budget*')


def test_measure_ini_reports(pytester: pytest.Pytester):
    pytester.makeini(f'[pytest]\nscriptlet_charm = {STEP_LIMIT}\nscriptlet_measure = true\n')
    pytester.makepyfile(
        RELATION_100
        + """
def test_it(scriptlet_ctx):
    assert scriptlet_ctx.run_raw(*relation_changed_100()).steps == 5545
"""
    )
    result = pytester.runpytest()
    result.assert_outcomes(passed=1)
    result.stdout.fnmatch_lines([
        '*scriptlet budget*',
        '*test_it: 5,545 of 100,000 steps (5.5%), 46,344 of 10,485,760 bytes (0.4%), '
        'relation_changed event',
    ])


@pytest.mark.parametrize(
    'args,ini,outcome',
    [
        (['--scriptlet-budget-fraction=0.05'], '', 'failed'),
        (['--scriptlet-budget-fraction=0.06'], '', 'passed'),
        ([], 'scriptlet_budget_fraction = 0.05\n', 'failed'),
        ([], 'scriptlet_budget_fraction = 0.5\n', 'passed'),
        # The option wins over the ini file.
        (['--scriptlet-budget-fraction=0.5'], 'scriptlet_budget_fraction = 0.05\n', 'passed'),
    ],
)
def test_budget_fraction(pytester: pytest.Pytester, args: list[str], ini: str, outcome: str):
    pytester.makeini(f'[pytest]\nscriptlet_charm = {STEP_LIMIT}\n{ini}')
    pytester.makepyfile(
        RELATION_100
        + """
def test_it(scriptlet_ctx):
    # The fraction turns measuring on.
    assert scriptlet_ctx.run_raw(*relation_changed_100()).measured
"""
    )
    result = pytester.runpytest(*args)
    result.assert_outcomes(**{outcome: 1})
    result.stdout.fnmatch_lines(['*5,545 of 100,000 steps (5.5%)*'])
    if outcome == 'failed':
        result.stdout.fnmatch_lines([
            '*scriptlet budget: over 5% of the limit*',
            '*relation_changed (event): 5,545*',
        ])


def test_budget_ignores_expected_failures(pytester: pytest.Pytester):
    data = TESTDATA / 'events' / 'relation_changed_2000.json'
    pytester.makeini(f'[pytest]\nscriptlet_charm = {STEP_LIMIT}\n')
    pytester.makepyfile(
        f"""
        import json, pathlib
        import pytest
        from scriptlet_testing import StepLimitError

        def test_too_big(scriptlet_ctx):
            data = json.loads(pathlib.Path({str(data)!r}).read_text())
            with pytest.raises(StepLimitError):
                scriptlet_ctx.run_raw(data['name'], data['attrs'])
        """
    )
    result = pytester.runpytest('--scriptlet-budget-fraction=0.5')
    result.assert_outcomes(passed=1)
    result.stdout.fnmatch_lines([
        '*110,046 of 100,000 steps (110.0%), 867,144 of 10,485,760 bytes (8.3%), '
        'relation_changed event, failed'
    ])


def test_budget_covers_contexts_made_in_the_test(pytester: pytest.Pytester):
    pytester.makepyfile(
        RELATION_100
        + f"""
from scriptlet_testing import Context

def test_it():
    with Context.from_charm({str(STEP_LIMIT)!r}, measure=True) as ctx:
        ctx.run_raw(*relation_changed_100())
"""
    )
    result = pytester.runpytest('--scriptlet-budget-fraction=0.01')
    result.assert_outcomes(failed=1)


def test_budget_checks_load_phase(pytester: pytest.Pytester):
    charm = pytester.path / 'charm'
    (charm / 'scriptlets').mkdir(parents=True)
    (charm / 'charmcraft.yaml').write_text('assumes: [unitless]\n')
    (charm / 'scriptlets' / 'hook.star').write_text(
        'def h(event):\n    pass\n\n'
        'def init():\n    x = [i for i in range(400)]\n    juju.observe("start_up", h)\n'
    )
    pytester.makeini('[pytest]\nscriptlet_charm = charm\n')
    pytester.makepyfile(
        """
        def test_it(scriptlet_ctx):
            scriptlet_ctx.run_raw('start_up', max_steps=10_000)
        """
    )
    result = pytester.runpytest('--scriptlet-budget-fraction=0.2')
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(['*start_up (load):*of 10,000 steps*'])


# Each call allocates about 2,500 bytes and takes 10 steps, so this uses far
# more of the allocation limit than of the step limit.
ALLOCATING_HOOK = """
def h(event):
    for i in range(3000):
        juju.set_status("active", "m")

def init():
    juju.observe("start_up", h)
"""


def _allocating_charm(pytester: pytest.Pytester) -> None:
    charm = pytester.path / 'charm'
    (charm / 'scriptlets').mkdir(parents=True)
    (charm / 'charmcraft.yaml').write_text('assumes: [unitless]\n')
    (charm / 'scriptlets' / 'hook.star').write_text(ALLOCATING_HOOK)
    pytester.makeini('[pytest]\nscriptlet_charm = charm\n')


@pytest.mark.parametrize('fraction,outcome', [('0.5', 'failed'), ('0.9', 'passed')])
def test_budget_checks_allocations(pytester: pytest.Pytester, fraction: str, outcome: str):
    _allocating_charm(pytester)
    pytester.makepyfile(
        """
        def test_it(scriptlet_ctx):
            out = scriptlet_ctx.run_raw('start_up')
            assert out.steps < 0.4 * scriptlet_ctx.max_steps
        """
    )
    result = pytester.runpytest(f'--scriptlet-budget-fraction={fraction}')
    result.assert_outcomes(**{outcome: 1})
    if outcome == 'failed':
        result.stdout.fnmatch_lines([
            '*scriptlet budget: over 50% of the limit*',
            '*start_up (event): * of 10,485,760 bytes (*%)*',
        ])
        # The steps are under the fraction, so they aren't listed as over it.
        result.stdout.no_fnmatch_line('*start_up (event): * steps*')


def test_budget_ignores_expected_alloc_failures(pytester: pytest.Pytester):
    _allocating_charm(pytester)
    pytester.makepyfile(
        """
        import pytest
        from scriptlet_testing import AllocLimitError

        def test_too_big(scriptlet_ctx):
            with pytest.raises(AllocLimitError):
                scriptlet_ctx.run_raw('start_up', max_allocs=1_000_000)
        """
    )
    result = pytester.runpytest('--scriptlet-budget-fraction=0.5')
    result.assert_outcomes(passed=1)
    result.stdout.fnmatch_lines(['*of 1,000,000 bytes*start_up event, failed'])


def test_budget_checks_load_allocations(pytester: pytest.Pytester):
    charm = pytester.path / 'charm'
    (charm / 'scriptlets').mkdir(parents=True)
    (charm / 'charmcraft.yaml').write_text('assumes: [unitless]\n')
    (charm / 'scriptlets' / 'hook.star').write_text(
        'def h(event):\n    pass\n\n'
        'def init():\n    x = [[i, i] for i in range(400)]\n    juju.observe("start_up", h)\n'
    )
    pytester.makeini('[pytest]\nscriptlet_charm = charm\n')
    pytester.makepyfile(
        """
        def test_it(scriptlet_ctx):
            scriptlet_ctx.run_raw('start_up', max_allocs=100_000)
        """
    )
    result = pytester.runpytest('--scriptlet-budget-fraction=0.1')
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(['*start_up (load):*of 100,000 bytes*'])


def test_user_properties_are_in_junit_xml(pytester: pytest.Pytester):
    pytester.makeini(f'[pytest]\nscriptlet_charm = {STEP_LIMIT}\nscriptlet_measure = true\n')
    pytester.makepyfile(
        RELATION_100
        + """
def test_it(scriptlet_ctx):
    scriptlet_ctx.run_raw(*relation_changed_100())
"""
    )
    pytester.runpytest('--junitxml=junit.xml', '-o', 'junit_family=xunit1')
    xml = (pytester.path / 'junit.xml').read_text()
    assert f'name="scriptlet_max_step_fraction" value="{5545 / 100_000}"' in xml
    assert f'name="scriptlet_max_alloc_fraction" value="{46344 / 10_485_760}"' in xml


def test_summary_when_steps_and_allocations_peak_in_different_phases(
    pytester: pytest.Pytester,
):
    charm = pytester.path / 'charm'
    (charm / 'scriptlets').mkdir(parents=True)
    (charm / 'charmcraft.yaml').write_text('assumes: [unitless]\n')
    # The event takes more steps than the load, but allocates less.
    (charm / 'scriptlets' / 'hook.star').write_text(
        'def h(event):\n    n = 0\n    for i in range(2000):\n        n = n + 0\n\n'
        'def init():\n    x = [[i, i] for i in range(400)]\n'
        '    juju.observe("start_up", h)\n'
    )
    pytester.makeini('[pytest]\nscriptlet_charm = charm\nscriptlet_measure = true\n')
    pytester.makepyfile(
        """
        def test_it(scriptlet_ctx):
            scriptlet_ctx.run_raw('start_up')
        """
    )
    result = pytester.runpytest()
    result.assert_outcomes(passed=1)
    result.stdout.fnmatch_lines(['*steps (*), start_up event; * bytes (*), start_up load'])


def test_observer_removed_after_each_test(pytester: pytest.Pytester):
    from scriptlet_testing import _context

    before = list(_context._run_observers)
    pytester.makeini(f'[pytest]\nscriptlet_charm = {SET_STATUS}\n')
    pytester.makepyfile(
        """
        from scriptlet_testing import _context

        def test_a(scriptlet_ctx):
            assert len(_context._run_observers) == {n}

        def test_b(scriptlet_ctx):
            assert len(_context._run_observers) == {n}
        """.replace('{n}', str(len(before) + 1))
    )
    pytester.runpytest('-p', 'no:randomly').assert_outcomes(passed=2)
    assert _context._run_observers == before


@pytest.mark.parametrize('value', ['0', '1.5', 'lots'])
def test_bad_fraction(pytester: pytest.Pytester, value: str):
    pytester.makeini(f'[pytest]\nscriptlet_budget_fraction = {value}\n')
    pytester.makepyfile('def test_it(): pass')
    result = pytester.runpytest()
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines(['*scriptlet*'])
