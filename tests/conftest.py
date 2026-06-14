import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _seed():
    np.random.seed(0)
