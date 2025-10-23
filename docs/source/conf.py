# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
import shutil
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, os.path.abspath('../../'))

# Import version from the package
from pypipr._version import __version__

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Pypipr'
copyright = '2025, S. Belgers'
author = 'S. Belgers'
release = __version__
version = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "myst_nb",
]

# Napoleon settings for Google/NumPy style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Autodoc settings
autodoc_default_options = { # type:ignore
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# Autosummary settings
autosummary_generate = True

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
}

# MyST parser settings
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_admonition",
    "html_image",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]

# MyST-NB (myst_nb) notebook execution settings
# Set notebook execution to 'off' to avoid running notebooks during documentation builds.
# See myst-nb docs for available options: execution_mode can be 'off', 'auto', 'force', 'cache', 'inline'.
nb_execution_mode = 'off'  # don't execute notebooks during build
nb_execution_allow_errors = True
nb_execution_timeout = 60

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Source file suffixes
source_suffix = {
    '.rst': None,
    '.md': 'myst_parser',
}

# The master toctree document
master_doc = 'index'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'includehidden': True,
    'titles_only': False
}

# Custom CSS and JS files
html_css_files = []
html_js_files = []

# HTML output options
html_title = f"{project} {version}"
html_short_title = project
html_favicon = None
html_logo = None

# Show source link
html_show_sourcelink = True
html_show_sphinx = True
html_show_copyright = True

# Output file base name for HTML help builder
htmlhelp_basename = 'pypiprdoc'

# -- Options for LaTeX output ------------------------------------------------

latex_elements = {}
latex_documents = [
    (master_doc, 'pypipr.tex', f'{project} Documentation', author, 'manual'),
]

# -- Options for manual page output ------------------------------------------

man_pages = [
    (master_doc, 'pypipr', f'{project} Documentation', [author], 1)
]

# -- Options for Texinfo output ----------------------------------------------

texinfo_documents = [
    (master_doc, 'pypipr', f'{project} Documentation', author, 'pypipr',
     'Package for processing pupil data, with a focus on the post illumination pupil response.', 'Miscellaneous'),
]

# -- Extension configuration -------------------------------------------------
