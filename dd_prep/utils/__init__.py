
"""dd_prep.utils

Make the utils folder importable and expose submodules.

This file dynamically imports all python modules in the current package
so that `from dd_prep.utils import <name>` works for modules placed
in this directory.
"""

from importlib import import_module
from pathlib import Path
import pkgutil

__all__ = []

# Determine the directory containing this file (the package folder)
package_path = Path(__file__).resolve().parent

# Iterate through all python modules found directly inside this folder.
# pkgutil.iter_modules yields tuples (finder, name, ispkg) for each module.
for finder, name, ispkg in pkgutil.iter_modules([str(package_path)]):
	# Skip private modules (those starting with an underscore).
	# These are usually implementation details the package does not expose.
	if name.startswith("_"):
		continue
	try:
		# Dynamically import the module as a submodule of this package
		# e.g. if name == 'helpers', this imports dd_prep.utils.helpers
		module = import_module(f"{__name__}.{name}")

		# Make the imported module available as an attribute of this package
		# so users can do: from dd_prep.utils import helpers
		globals()[name] = module

		# Record the public name so `from dd_prep.utils import *` knows about it
		__all__.append(name)
	except Exception:
		# If a module raises during import, ignore it so the package
		# itself can still be imported. This prevents one broken helper
		# from making the whole utils package unusable.
		pass

# Expose package version if defined in a _version.py or version variable.
# This allows users to check dd_prep.utils.__version__ if present.
try:
	from ._version import __version__  # type: ignore
except Exception:
	# Default to a placeholder version if no _version.py exists or
	# if importing it failed for any reason.
	__version__ = "0.0.0"
