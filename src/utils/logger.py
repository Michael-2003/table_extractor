import time
from contextlib import contextmanager


@contextmanager
def timed(msg: str):
    t0 = time.time()
    yield
    dt = time.time() - t0
    print(f"[timed] {msg}: {dt:.3f}s")