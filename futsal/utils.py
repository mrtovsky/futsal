import contextlib

import numpy as np


@contextlib.contextmanager
def set_local_seed(seed):
    """Localy sets a fixed numpy seed."""
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)
