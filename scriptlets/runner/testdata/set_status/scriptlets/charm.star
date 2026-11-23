load("lib/checks.star", "port_problem")

def on_config_changed(event):
    debug("port is a", type(event.port), "and ratio is a", type(event.ratio))
    problem = port_problem(event.port)
    if problem:
        juju.set_status("blocked", problem)
        return
    print("serving on", event.port)
    juju.set_status("active", message="serving on %d" % event.port)

def on_update_status(event):
    juju.set_status("active")

def init():
    juju.observe("config_changed", on_config_changed)
    juju.observe("update_status", on_update_status)
