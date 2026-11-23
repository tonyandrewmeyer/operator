"""Pluggable execution boundaries for the add-reproducer pipeline.

Two seams, each with a live implementation (never exercised by the test
suite -- nothing in the test environment can reach OpenRouter or a juju controller)
and a fixture/replay implementation (what every test actually runs against):

- `llm`: LLM calls (OpenRouter), for hypothesis extraction and surface
  inference.
- `runner`: the juju/charmcraft reproduction runner.
"""
