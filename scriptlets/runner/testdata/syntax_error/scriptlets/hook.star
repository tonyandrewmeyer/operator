def on_config_changed(event):
    n = 0
    while n < 3:
        n += 1

def init():
    juju.observe("config_changed", on_config_changed)
