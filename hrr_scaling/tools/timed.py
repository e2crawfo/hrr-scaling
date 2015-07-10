import time

def timer(func):
    def timed(*args, **kwargs):
        start = time.time()
        ret = func(*args, **kwargs)
        end = time.time()
        print "Time:", end - start
        return ret
    return timed

def namedtimer(name=None):
    def decorator(func):
        def timed(*args, **kwargs):
            start = time.time()
            ret = func(*args, **kwargs)
            end = time.time()
            print "Time for", name, ":", end - start
            return ret
        return timed
    return decorator

