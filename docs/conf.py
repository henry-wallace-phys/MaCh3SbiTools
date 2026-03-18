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
# Suppress warnings from third-party packages
suppress_warnings = [
    "ref.ref",  # undefined labels in torch/rich docstrings
    "ref.doc",
    "docutils",
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

autodoc_default_options = {
    "imported-members": False,
}
autodoc_type_aliases = {
    "Style": "rich.style.Style",
}


def autodoc_skip_member(app, what, name, obj, skip, options):
    """Skip members not defined in mach3sbitools."""
    module = getattr(obj, "__module__", "") or ""
    if module and not module.startswith("mach3sbitools"):
        return True
    return skip


def setup(app):
    app.connect("autodoc-skip-member", autodoc_skip_member)
