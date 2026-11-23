def count_bad(units):
    bad = 0
    for unit in units:
        for key in sorted(unit.keys()):
            if not unit[key].startswith("http://"):
                bad += 1
    return bad

def on_relation_changed(event):
    bad = count_bad(event.units)
    juju.set_status("active" if bad == 0 else "blocked", "%d bad of %d" % (bad, len(event.units)))

def init():
    juju.observe("relation_changed", on_relation_changed)
