import subprocess
import sys

try:
  from dunamai import Version
except ImportError:
  subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'dunamai'])
  from dunamai import Version  # type: ignore[import]

try:
  __version__ = Version.from_any_vcs().serialize()
except RuntimeError:
  __version__ = '0.0.0'
except ValueError as e:
  if "time data '%cI' does not match format '%Y-%m-%dT%H:%M:%S%z'" in str(e):
    __version__ = '0.0.0'
  else:
    raise
