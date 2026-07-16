(manage-resources)=
# How to manage resources

> See also: {external+juju:ref}`Juju | Charm resource <charm-resource>`, {external+juju:ref}`Juju | Manage charm resources <manage-charm-resources>`, {external+charmcraft:ref}`Charmcraft | Manage resources <manage-resources>`

## Implement the feature

Resources are declared in your charm's `charmcraft.yaml`. To use one in your charm, fetch its on-disk path from the model and read the file as normal.

For example, suppose your `charmcraft.yaml` contains this resource definition:

```yaml
resources:
  my-resource:
    type: file
    filename: somefile.txt
    description: test resource
```

In `src/charm.py`, use [`Model.resources.fetch()`](ops.Resources.fetch) to get the path to the resource, then read it as you would any other file:

```python
import logging
import ops

logger = logging.getLogger(__name__)


class MyCharm(ops.CharmBase):
    def _on_config_changed(self, event: ops.ConfigChangedEvent):
        try:
            resource_path = self.model.resources.fetch('my-resource')
        except ops.ModelError:
            self.unit.status = ops.BlockedStatus(
                "Couldn't fetch resource 'my-resource'; run `juju debug-log` for more info."
            )
            logger.exception("Couldn't fetch resource 'my-resource'.")
            return
        except NameError:
            self.unit.status = ops.BlockedStatus(
                "Resource 'my-resource' not found; is it declared in charmcraft.yaml?"
            )
            logger.exception("Resource 'my-resource' not declared in charmcraft.yaml.")
            return

        with resource_path.open() as f:
            content = f.read()
        # Use `content` as needed.
```

[`fetch()`](ops.Resources.fetch) returns a [`pathlib.Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path) on success. It raises [`NameError`](https://docs.python.org/3/library/exceptions.html#NameError) if the resource isn't declared in `charmcraft.yaml`, and [`ops.ModelError`](ops.ModelError) if Juju can't provide the resource — for example, when deploying from a local charm file without passing `--resource`.

During development, attach a local file at deploy time to iterate without republishing the charm:

```text
echo "TEST" > /tmp/somefile.txt
charmcraft pack
juju deploy ./my-charm.charm --resource my-resource=/tmp/somefile.txt
```

## Test the feature

> See first: {ref}`write-unit-tests-for-a-charm`

Make resources available to the charm through [`State.resources`](ops.testing.State.resources). Each entry is a [`testing.Resource`](ops.testing.Resource) pointing at a local file that stands in for the real resource content:

```python
import pathlib

from ops import testing

ctx = testing.Context(
    MyCharm, meta={'name': 'my-charm', 'resources': {'my-resource': {'type': 'file', 'filename': 'somefile.txt'}}}
)
resource = testing.Resource(name='my-resource', path='/path/to/somefile.txt')
with ctx(ctx.on.config_changed(), testing.State(resources={resource})) as mgr:
    path = mgr.charm.model.resources.fetch('my-resource')
    assert path == pathlib.Path('/path/to/somefile.txt')
```
