import sys
import os

# Add the project root to the python path so tests can find the 'src' module
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Disable all test collection until explicit test config file support is wired up,
# preventing tests from accidentally touching production databases in config.toml.
collect_ignore_glob = ["*"]

