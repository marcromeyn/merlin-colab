from collections import defaultdict
from pathlib import Path

import pytest


@pytest.fixture
def tmpdir():
    tmp = Path("./tmp")
    tmp.mkdir(exist_ok=True)
    return tmp
