# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from typing import Any, get_origin

import numpy as np
from numpy.typing import ArrayLike

sys.path.insert(0, os.path.abspath('../../'))

from piprkit._version import __version__

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'piprkit'
copyright = '2025–2026, S. Belgers'
author = 'S. Belgers'
release = __version__
version = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "myst_nb",
]

napoleon_include_special_with_doc = True
napoleon_use_ivar = True

def typehints_formatter(annotation: Any, config: Any) -> str | None:
    if get_origin(annotation) is np.ndarray:
        return ":class:`numpy.ndarray`"
    if annotation == ArrayLike:
        return ":term:`numpy:array_like`"
    return None

def _skip_builtin_members(app: Any, what: str, name: str, obj: Any, skip: bool, options: Any) -> bool | None:
    if getattr(obj, "__objclass__", None) in (float, object):
        return True
    owner = getattr(obj, "__self__", None)
    if isinstance(owner, type) and issubclass(owner, float) and name in vars(float):
        return True
    return None

def setup(app: Any) -> None:
    app.connect("autodoc-skip-member", _skip_builtin_members)

autodoc_type_aliases = {"ArrayLike": "numpy.typing.ArrayLike"}

autodoc_default_options = { # type:ignore
    'members': True,
    'member-order': 'bysource',
    'undoc-members': True,
    'inherited-members': True,
    'exclude-members': '__weakref__'
}

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
}

myst_enable_extensions = [
    "dollarmath",
]

# MyST-NB (myst_nb) notebook execution settings
# See myst-nb docs for available options: execution_mode can be 'off', 'auto', 'force', 'cache', 'inline'.
nb_execution_mode = 'force'
nb_execution_raise_on_error = True
nb_execution_timeout = 60
nb_render_markdown_format = "myst"

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

source_suffix: dict[str, Any] = {
    ".rst": None,
    ".ipynb": "myst-nb",
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_theme_options: dict[str, Any] = {
    'collapse_navigation': False,
}
html_title = f"{project} {version}"
