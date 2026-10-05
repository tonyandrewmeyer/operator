# The `ops` library

![CI Status](https://github.com/canonical/operator/actions/workflows/framework-tests.yaml/badge.svg)

The `ops` library is a Python framework for developing and testing Kubernetes and machine [charms](https://charmhub.io/). While charms can be written in any language, `ops` defines the latest standard, and charmers are encouraged to use Python with `ops` for all charms. The library is an official component of the Charm SDK, itself a part of [the Juju universe](https://canonical.com/juju).

> - `ops` is  [available on PyPI](https://pypi.org/project/ops/).
> - The latest version of `ops` requires Python 3.10 or above.
> - Read our [docs](https://canonical.com/juju/docs/ops/latest/) for tutorials, how-to guides, the library reference, and more.

## A minimal charm

A charm is a Python class that observes Juju events. This one sets the unit status to active when the unit starts:

```python
import ops


class MyCharm(ops.CharmBase):
    def __init__(self, framework: ops.Framework):
        super().__init__(framework)
        framework.observe(self.on.start, self._on_start)

    def _on_start(self, event: ops.StartEvent):
        self.unit.status = ops.ActiveStatus('ready')


if __name__ == '__main__':
    ops.main(MyCharm)
```

You can test the charm without Juju, using `ops.testing`:

```python
from ops import testing

from charm import MyCharm


def test_start():
    ctx = testing.Context(MyCharm)
    state_out = ctx.run(ctx.on.start(), testing.State())
    assert state_out.unit_status == testing.ActiveStatus('ready')
```

## Try it out

1. Install [Concierge](https://github.com/canonical/concierge) and use it to set up a Juju development environment:

   ```shell
   sudo snap install concierge --classic
   sudo concierge prepare --preset k8s
   ```

2. Install [Charmcraft](https://documentation.ubuntu.com/charmcraft/) and create a Kubernetes charm that uses `ops`:

   ```shell
   sudo snap install charmcraft --classic
   mkdir my-charm && cd my-charm
   charmcraft init --profile kubernetes
   ```

   The generated project includes `src/charm.py`, unit tests in `tests/unit`, and integration tests in `tests/integration`.

3. Run the unit tests, then pack and deploy the charm:

   ```shell
   tox -e unit
   charmcraft pack
   juju deploy ./*.charm --resource httpbin-image=kennethreitz/httpbin
   ```

For a full walkthrough, follow the [Kubernetes charm tutorial](https://canonical.com/juju/docs/ops/latest/tutorial/from-zero-to-hero-write-your-first-kubernetes-charm/). When you're finished, tear down the environment with `sudo concierge restore`.

## Next steps

- Read the [docs](https://canonical.com/juju/docs/ops/latest/).
- Read our [Code of conduct](https://ubuntu.com/community/code-of-conduct) and join our [chat](https://matrix.to/#/#charmhub-ops:ubuntu.com) and [forum](https://discourse.charmhub.io/) or [open an issue](https://github.com/canonical/operator/issues).
- Read our [contributing guide](https://github.com/canonical/operator/blob/main/CONTRIBUTING.md) and contribute!
