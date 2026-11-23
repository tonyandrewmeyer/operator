def port_problem(port):
    if type(port) != "int":
        return "port must be an int, got %s" % type(port)
    if port < 1 or port > 65535:
        return "port %d out of range" % port
    return None
