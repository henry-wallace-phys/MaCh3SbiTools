import os
import sys
from importlib.metadata import version as get_version

sys.path.insert(0, os.path.abspath("../src"))

project = "mach3sbitools"
author = "Henry Wallace"
release = get_version("mach3sbitools")
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
]

html_theme = "furo"

# Napoleon — Sphinx-style docstrings
napoleon_google_docstring = False
napoleon_numpy_docstring = False
napoleon_use_sphinx_docstring = True

# Autodoc
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_typehints_format = "short"
add_module_names = False
