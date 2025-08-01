(integrate-your-charm-with-postgresql)=
# Integrate your charm with PostgreSQL

<!-- UPDATE LINKS

Please add a link to `fetch-lib` documentation in the charmcraft docs, and maybe in the observers section a link to information about custom events (Juju docs?).

-->

> <small> {ref}`From Zero to Hero: Write your first Kubernetes charm <from-zero-to-hero-write-your-first-kubernetes-charm>` > Integrate your charm with PostgreSQL</small>
>
> **See previous: {ref}`Make your charm configurable <make-your-charm-configurable>`**

````{important}

This document is part of a  series, and we recommend you follow it in sequence.  However, you can also jump straight in by checking out the code from the previous chapter:

```text
git clone https://github.com/canonical/operator.git
cd operator/examples/k8s-2-configurable
```

````

A charm often requires or supports relations to other charms. For example, to make our application fully functional we need to connect it to the PostgreSQL database. In this chapter of the tutorial we will update our charm so that it can be integrated with the existing [PostgreSQL charm](https://charmhub.io/postgresql-k8s?channel=14/stable).

## Fetch the required database interface charm libraries

In `charmcraft.yaml`, add a `charm-libs` section before the `containers` section:

```yaml
charm-libs:
  - lib: data_platform_libs.data_interfaces
    version: "0"
```

This tells Charmcraft that your charm requires the [data_interfaces](https://charmhub.io/data-platform-libs/libraries/data_interfaces) charm library from Charmhub.

Next, run the following command to download the library:

```text
ubuntu@charm-dev:~/fastapi-demo$ charmcraft fetch-libs
```

Your charm directory should now contain the structure below:

```text
lib
└── charms
    └── data_platform_libs
        └── v0
            └── data_interfaces.py
```

Well done, you've got everything you need to set up a database relation!

## Define the charm relation interface

Now, time to define the charm relation interface.

First, find out the name of the interface that PostgreSQL offers for other charms to connect to it. According to the [documentation of the PostgreSQL charm](https://charmhub.io/postgresql-k8s?channel=14/stable), the interface is called `postgresql_client`.

Next, open the `charmcraft.yaml` file of your charm and, before the `charm-libs` section, define a relation endpoint using a `requires` block, as below. This endpoint says that our charm is requesting a relation called `database` over an interface called `postgresql_client` with a maximum number of supported connections of 1. (Note: Here, `database` is a custom relation name, though in general we recommend sticking to default recommended names for each charm.)

```yaml
requires:
  database:
    interface: postgresql_client
    limit: 1
    optional: false
```

That will tell `juju` that our charm can be integrated with charms that provide the same `postgresql_client` interface, for example, the official PostgreSQL charm.

Import the database interface libraries and define database event handlers

We now need to implement the logic that wires our application to a database. When a relation between our application and the data platform is formed, the provider side (that is: the data platform) will create a database for us and it will provide us with all the information we need to connect to it over the relation -- for example, username, password, host, port, and so on. On our side, we nevertheless still need to set the relevant environment variables to point to the database and restart the service.

To do so, we need to update our charm `src/charm.py` to do all of the following:

* Import the `DataRequires` class from the interface library; this class represents the relation data exchanged in the client-server communication.
* Define the event handlers that will be called during the relation lifecycle.
* Bind the event handlers to the observed relation events.

### Import the database interface libraries

First, at the top of the file, import the database interfaces library:

```python
# Import the 'data_interfaces' library.
# The import statement omits the top-level 'lib' directory
# because 'charmcraft pack' copies its contents to the project root.
from charms.data_platform_libs.v0.data_interfaces import DatabaseCreatedEvent, DatabaseRequires
```

````{important}

You might have noticed that despite the charm library being placed in the `lib/charms/...`, we are importing it via:

```python
from charms.data_platform_libs ...
```

and not

```python
from lib.charms.data_platform_libs...
```

The former is not resolvable by default but everything works fine when the charm is deployed. Why? Because the `dispatch` script in the packed charm sets the `PYTHONPATH` environment variable to include the `lib` directory when it executes your `src/charm.py` code. This tells Python it can check the `lib` directory when looking for modules and packages at import time.

If you're experiencing issues with your IDE or just trying to run the `charm.py` file on your own, make sure to set/update `PYTHONPATH` to include `lib` directory as well.

```bash
# from the charm project directory (~/fastapi-demo), set
export PYTHONPATH=lib
# or update
export PYTHONPATH=lib:$PYTHONPATH
```

````

### Add relation event observers

Next, in the `__init__` method, define a new instance of the 'DatabaseRequires' class. This is required to set the right permissions scope for the PostgreSQL charm. It will create a new user with a password and a database with the required name (below, `names_db`), and limit the user permissions to only this particular database (that is, below, `names_db`).


```python
# The 'relation_name' comes from the 'charmcraft.yaml file'.
# The 'database_name' is the name of the database that our application requires.
self.database = DatabaseRequires(self, relation_name='database', database_name='names_db')
```

Now, add event observers for all the database events:

```python
# See https://charmhub.io/data-platform-libs/libraries/data_interfaces
framework.observe(self.database.on.database_created, self._on_database_created)
framework.observe(self.database.on.endpoints_changed, self._on_database_created)
```

### Fetch the database authentication data

Now we need to extract the database authentication data and endpoints information. We can do that by adding a `fetch_postgres_relation_data` method to our charm class. Inside this method, we first retrieve relation data from the PostgreSQL using the `fetch_relation_data`  method of the `database` object. We then log the retrieved data for debugging purposes. Next we process any non-empty data to extract endpoint information, the username, and the password and return this process data as a dictionary. Finally, we ensure that, if no data is retrieved, we return an empty dictionary, so that the caller knows that the database is not yet ready.

```python
def fetch_postgres_relation_data(self) -> dict[str, str]:
    """Fetch postgres relation data.

    This function retrieves relation data from a postgres database using
    the `fetch_relation_data` method of the `database` object. The retrieved data is
    then logged for debugging purposes, and any non-empty data is processed to extract
    endpoint information, username, and password. This processed data is then returned as
    a dictionary. If no data is retrieved, the unit is set to waiting status and
    the program exits with a zero status code.
    """
    relations = self.database.fetch_relation_data()
    logger.debug('Got following database data: %s', relations)
    for data in relations.values():
        if not data:
            continue
        logger.info('New database endpoint is %s', data['endpoints'])
        host, port = data['endpoints'].split(':')
        db_data = {
            'db_host': host,
            'db_port': port,
            'db_username': data['username'],
            'db_password': data['password'],
        }
        return db_data
    return {}
```

### Share the authentication information with your application

Our application consumes database authentication information in the form of environment variables. Let's update the Pebble service definition with an `environment` key and let's set this key to a dynamic value. Update the `_update_layer_and_restart()` method to read in the environment and pass it in when creating the Pebble layer:

```python
def _update_layer_and_restart(self) -> None:
    """Define and start a workload using the Pebble API.

    You'll need to specify the right entrypoint and environment
    configuration for your specific workload. Tip: you can see the
    standard entrypoint of an existing container using docker inspect
    Learn more about interacting with Pebble at
        https://documentation.ubuntu.com/ops/latest/reference/pebble.html
    Learn more about Pebble layers at
        https://documentation.ubuntu.com/pebble/how-to/use-layers/
    """
    # Learn more about statuses at
    # https://documentation.ubuntu.com/juju/3.6/reference/status/
    self.unit.status = ops.MaintenanceStatus('Assembling Pebble layers')
    try:
        config = self.load_config(FastAPIConfig)
    except ValueError as e:
        logger.error('Configuration error: %s', e)
        return
    env = self.get_app_environment()
    try:
        self.container.add_layer(
            'fastapi_demo',
            self._get_pebble_layer(config.server_port, env),
            combine=True,
        )
        logger.info("Added updated layer 'fastapi_demo' to Pebble plan")

        # Tell Pebble to incorporate the changes, including restarting the
        # service if required.
        self.container.replan()
        logger.info(f"Replanned with '{self.pebble_service_name}' service")
    except (ops.pebble.APIError, ops.pebble.ConnectionError) as e:
        logger.info('Unable to connect to Pebble: %s', e)
```

We've also removed three `self.unit.status = ` lines. We'll handle replacing those shortly.

Now, update your `_get_pebble_layer()` method to use the passed environment:

```python
def _get_pebble_layer(self, port: int, environment: dict[str, str]) -> ops.pebble.Layer:
    """A Pebble layer for the FastAPI demo services."""
    command = ' '.join([
        'uvicorn',
        'api_demo_server.app:app',
        '--host=0.0.0.0',
        f'--port={port}',
    ])
    pebble_layer: ops.pebble.LayerDict = {
        'summary': 'FastAPI demo service',
        'description': 'pebble config layer for FastAPI demo server',
        'services': {
            self.pebble_service_name: {
                'override': 'replace',
                'summary': 'fastapi demo',
                'command': command,
                'startup': 'enabled',
                'environment': environment,
            }
        },
    }
    return ops.pebble.Layer(pebble_layer)
```

Now, let's define this method such that, every time it is called, it dynamically fetches database authentication data and also prepares the output in a form that our application can consume, as below:

```python
def get_app_environment(self) -> dict[str, str]:
    """Prepare environment variables for the application.

    This property method creates a dictionary containing environment variables
    for the application. It retrieves the database authentication data by calling
    the `fetch_postgres_relation_data` method and uses it to populate the dictionary.
    If any of the values are not present, it will be set to None.
    The method returns this dictionary as output.
    """
    db_data = self.fetch_postgres_relation_data()
    if not db_data:
        return {}
    env = {
        key: value
        for key, value in {
            'DEMO_SERVER_DB_HOST': db_data.get('db_host', None),
            'DEMO_SERVER_DB_PORT': db_data.get('db_port', None),
            'DEMO_SERVER_DB_USER': db_data.get('db_username', None),
            'DEMO_SERVER_DB_PASSWORD': db_data.get('db_password', None),
        }.items()
        if value is not None
    }
    return env
```

Finally, let's define the method that is called on the database created event:

```python
def _on_database_created(self, _: DatabaseCreatedEvent) -> None:
    """Event is fired when postgres database is created."""
    self._update_layer_and_restart()
```

The diagram below illustrates the workflow for the case where the database relation exists and for the case where it does not:

![Integrate your charm with PostgreSQL](../../resources/integrate_your_charm_with_postgresql.png)

## Update the unit status to reflect the relation state

Now that the charm is getting more complex, there are many more cases where the unit status needs to be set. It's often convenient to do this in a more declarative fashion, which is where the collect-status event can be used.

> Read more: [](ops.CollectStatusEvent)

In your charm's `__init__` add a new observer:

```python
# Report the unit status after each event.
framework.observe(self.on.collect_unit_status, self._on_collect_status)
```

And define a method that does the various checks, adding appropriate statuses. The library will take care of selecting the 'most significant' status for you.

```python
def _on_collect_status(self, event: ops.CollectStatusEvent) -> None:
    try:
        self.load_config(FastAPIConfig)
    except ValueError as e:
        event.add_status(ops.BlockedStatus(str(e)))
    if not self.model.get_relation('database'):
        # We need the user to do 'juju integrate'.
        event.add_status(ops.BlockedStatus('Waiting for database relation'))
    elif not self.database.fetch_relation_data():
        # We need the charms to finish integrating.
        event.add_status(ops.WaitingStatus('Waiting for database relation'))
    try:
        status = self.container.get_service(self.pebble_service_name)
    except (ops.pebble.APIError, ops.pebble.ConnectionError, ops.ModelError):
        event.add_status(ops.MaintenanceStatus('Waiting for Pebble in workload container'))
    else:
        if not status.is_running():
            event.add_status(ops.MaintenanceStatus('Waiting for the service to start up'))
    # If nothing is wrong, then the status is active.
    event.add_status(ops.ActiveStatus())
```

We also want to clean up the code to remove the places where we're setting the status outside of this method, other than anywhere we're wanting a status to show up *during* the event execution (such as `MaintenanceStatus`). If you missed doing so above, in `_update_layer_and_restart`, remove the lines:

```python
self.unit.status = ops.ActiveStatus()
```

```python
self.unit.status = ops.MaintenanceStatus('Waiting for Pebble in workload container')
```

```python
self.unit.status = ops.BlockedStatus(str(e))
```

## Validate your charm

Time to check the results!

First, repack and refresh your charm:

```text
charmcraft pack
juju refresh \
  --path="./demo-api-charm_ubuntu-22.04-amd64.charm" \
  demo-api-charm --force-units --resource \
  demo-server-image=ghcr.io/canonical/api_demo_server:1.0.1
```

Next, deploy the `postgresql-k8s` charm:

```text
juju deploy postgresql-k8s --channel=14/stable --trust
```

Now,  integrate our charm with the newly deployed `postgresql-k8s` charm:

```text
juju integrate postgresql-k8s demo-api-charm
```

> Read more: {external+juju:ref}`Juju | Relation (integration) <relation>`, [`juju integrate`](inv:juju:std:label#command-juju-integrate)

Finally, run:

```text
juju status --relations --watch 1s
```

You should see both applications get to the `active` status, and also that the `postgresql-k8s` charm has a relation to the `demo-api-charm` over the `postgresql_client` interface, as below:

```text
Model        Controller  Cloud/Region        Version  SLA          Timestamp
welcome-k8s  microk8s    microk8s/localhost  3.6.8    unsupported  13:50:39+01:00

App             Version  Status  Scale  Charm           Channel    Rev  Address         Exposed  Message
demo-api-charm           active      1  demo-api-charm               2  10.152.183.233  no
postgresql-k8s  14.15    active      1  postgresql-k8s  14/stable  495  10.152.183.195  no

Unit               Workload  Agent  Address      Ports  Message
demo-api-charm/0*  active    idle   10.1.157.90
postgresql-k8s/0*  active    idle   10.1.157.92         Primary

Integration provider           Requirer                       Interface          Type     Message
postgresql-k8s:database        demo-api-charm:database        postgresql_client  regular
postgresql-k8s:database-peers  postgresql-k8s:database-peers  postgresql_peers   peer
postgresql-k8s:restart         postgresql-k8s:restart         rolling_op         peer
postgresql-k8s:upgrade         postgresql-k8s:upgrade         upgrade            peer
```

The relation appears to be up and running, but we should also test that it's working as intended. First, let's try to write something to the database by posting some name to the database via API using `curl` as below -- where `10.1.157.90` is a pod IP and `8000` is our app port. You can repeat the command for multiple names.

```text
curl -X 'POST' \
  'http://10.1.157.90:8000/addname/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/x-www-form-urlencoded' \
  -d 'name=maksim'
```

```{important}

If you changed the `server-port` config value in the previous section, don't forget to change it back to 8000 before doing this!
```

Second, let's try to read something from the database by running:

```text
curl 10.1.157.90:8000/names
```

This should produce something similar to the output below (of course, with the names that *you* decided to use):

```text
{"names":{"1":"maksim","2":"simon"}}
```

Congratulations, your relation with PostgreSQL is functional!

## Write unit tests

Now that our charm uses `fetch_postgres_relation_data` to extract database authentication data and endpoint information from the relation data, we should write a test for the feature. Here, we're not testing the `fetch_postgres_relation_data` function directly, but rather, we're checking that the response to a Juju event is what it should be:

```python
def test_relation_data():
    ctx = testing.Context(FastAPIDemoCharm)
    relation = testing.Relation(
        endpoint='database',
        interface='postgresql_client',
        remote_app_name='postgresql-k8s',
        remote_app_data={
            'endpoints': 'example.com:5432',
            'username': 'foo',
            'password': 'bar',
        },
    )
    container = testing.Container(name='demo-server', can_connect=True)
    state_in = testing.State(
        containers={container},
        relations={relation},
        leader=True,
    )

    state_out = ctx.run(ctx.on.relation_changed(relation), state_in)

    assert state_out.get_container(container.name).layers['fastapi_demo'].services[
        'fastapi-service'
    ].environment == {
        'DEMO_SERVER_DB_HOST': 'example.com',
        'DEMO_SERVER_DB_PORT': '5432',
        'DEMO_SERVER_DB_USER': 'foo',
        'DEMO_SERVER_DB_PASSWORD': 'bar',
    }
```

In this chapter, we also defined a new method `_on_collect_status` that checks various things, including whether the required database relation exists. If the relation doesn't exist, we wait and set the unit status to `blocked`. We can also add a test to cover this behaviour:

```python
def test_no_database_blocked():
    ctx = testing.Context(FastAPIDemoCharm)
    container = testing.Container(name='demo-server', can_connect=True)
    state_in = testing.State(
        containers={container},
        leader=True,
    )  # We've omitted relation data from the input state.

    state_out = ctx.run(ctx.on.collect_unit_status(), state_in)

    assert state_out.unit_status == testing.BlockedStatus('Waiting for database relation')
```

Then modify `test_pebble_layer`. Since `test_pebble_layer` doesn't arrange a database relation, the unit will be in `blocked` status instead of `active`. Replace the `assert state_out.unit_status` line by:

```python
    # Check the unit is blocked:
    assert state_out.unit_status == testing.BlockedStatus('Waiting for database relation')
```

Now run `tox -e unit` to make sure all test cases pass.

## Write an integration test

Now that our charm integrates with the PostgreSQL database, if there's not a database relation, the app will be in `blocked` status instead of `active`. Let's tweak our existing integration test `test_build_and_deploy` accordingly, setting the expected status as `blocked` in `ops_test.model.wait_for_idle`:

```python
@pytest.mark.abort_on_fail
async def test_build_and_deploy(ops_test: OpsTest):
    """Build the charm-under-test and deploy it together with related charms.

    Assert on the unit status before integration or configuration.
    """
    # Build and deploy charm from local source folder
    charm = await ops_test.build_charm('.')
    resources = {
        'demo-server-image': METADATA['resources']['demo-server-image']['upstream-source']
    }

    # Deploy the charm and wait for blocked/idle status.
    # The app will not be in active status as this requires a database relation.
    await asyncio.gather(
        ops_test.model.deploy(charm, resources=resources, application_name=APP_NAME),
        ops_test.model.wait_for_idle(
            apps=[APP_NAME], status='blocked', raise_on_blocked=False, timeout=300
        ),
    )
```

Then, let's add another test case to check the integration is successful. For that, we need to deploy a database to the test cluster and integrate both applications. If everything works as intended, the charm should report an active status.

In your `tests/integration/test_charm.py` file add the following test case:

```python
@pytest.mark.abort_on_fail
async def test_database_integration(ops_test: OpsTest):
    """Verify that the charm integrates with the database.

    Assert that the charm is active if the integration is established.
    """
    await ops_test.model.deploy(
        application_name='postgresql-k8s',
        entity_url='postgresql-k8s',
        channel='14/stable',
    )
    await ops_test.model.integrate(f'{APP_NAME}', 'postgresql-k8s')
    await ops_test.model.wait_for_idle(
        apps=[APP_NAME], status='active', raise_on_blocked=False, timeout=300
    )
```

```{important}

If you run one test and then the other as separate `pytest ...` invocations, then two separate models will be created unless you pass `--model=some-existing-model` to inform pytest-operator to use a model you provide.

```

In your Multipass Ubuntu VM, run the test again:

```text
ubuntu@charm-dev:~/fastapi-demo$ tox -e integration
```

The test may again take some time to run.

```{tip}

To make things faster, use the `--model=<existing model name>` to inform `pytest-operator` to use the model it has created for the first test. Otherwise, charmers often have a way to cache their pack or deploy results.

```

When it's done, the output should show two passing tests:

```text
...
INFO     pytest_operator.plugin:plugin.py:621 Using tmp_path: /home/ubuntu/fastapi-demo/.tox/integration/tmp/pytest/test-charm-l5a20
INFO     pytest_operator.plugin:plugin.py:1213 Building charm demo-api-charm
INFO     pytest_operator.plugin:plugin.py:1218 Built charm demo-api-charm in 34.47s
INFO     juju.model:__init__.py:3254 Waiting for model:
  demo-api-charm (missing)
INFO     juju.model:__init__.py:2301 Deploying local:demo-api-charm-0
INFO     juju.model:__init__.py:3254 Waiting for model:
  demo-api-charm/0 [idle] blocked: Waiting for database relation
PASSED
tests/integration/test_charm.py::test_database_integration
--------------------------------------------------------------------------------------- live log call ----------------------------------------------------------------------------------------
INFO     juju.model:__init__.py:2301 Deploying ch:amd64/jammy/postgresql-k8s-495
INFO     juju.model:__init__.py:3254 Waiting for model:
  demo-api-charm/0 [idle] blocked: Waiting for database relation
PASSED
...
```

Congratulations, with this integration test you have verified that your charm's relation to PostgreSQL works as well!

## Review the final code

For the full code,  see [our example charm for this chapter](https://github.com/canonical/operator/tree/main/examples/k8s-3-postgresql).

> **See next: {ref}`Expose your charm's operational tasks via actions <expose-operational-tasks-via-actions>`**
