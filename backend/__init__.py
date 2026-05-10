"""ReproLab Agent backend package."""

# Load .env into os.environ before any submodule imports a provider that
# reads API keys from os.environ directly (Hermes audit providers, the
# Runpod backend's env-var fallback, etc.). pydantic-settings handles our
# REPROLAB_* Settings fields separately, but it does not populate
# os.environ — so plain os.environ.get() reads would return None without
# this bootstrap. See backend/_env_bootstrap.py.
from backend._env_bootstrap import load_dotenv_once as _load_dotenv_once

_load_dotenv_once()

__version__ = "0.1.0"
