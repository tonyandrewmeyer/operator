def on_config_changed(event):
    juju.set_status("active", "ok", "extra")

def init():
    juju.observe("config_changed", on_config_changed)
