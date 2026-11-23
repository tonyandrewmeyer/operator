def endpoint(config):
    return "http://%s:%d" % (config["host"], config["port"])

def on_config_changed(event):
    print("config changed")
    juju.set_status("maintenance", "configuring")
    juju.set_status("active", endpoint(event.config))

def init():
    juju.observe("config_changed", on_config_changed)
