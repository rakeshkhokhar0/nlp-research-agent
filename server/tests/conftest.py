import os
import sys
from pathlib import Path

import pytest

# Ensure server app is on sys.path
SERVER_ROOT = Path(__file__).resolve().parents[1]
if str(SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVER_ROOT))


@pytest.fixture(scope="session")
def model_metadata():
    """Provides Base metadata with all registered models."""
    from app.database.model_registry import Base

    return Base.metadata

