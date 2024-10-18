# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../..'))

_cwd = os.path.abspath(os.path.expanduser(os.path.dirname(__file__)))


def abspath(rel: str) -> str:
    """
    Take paths relative to the current file and
    convert them to absolute paths.

    Parameters
    ------------
    rel : str
      Relative path, IE '../stuff'

    Returns
    -------------
    abspath : str
      Absolute path, IE '/home/user/stuff'
    """
    # current working directory
    return os.path.abspath(os.path.join(_cwd, rel))


# -- Project information -----------------------------------------------------

project = 'FLAP'
copyright = '2019-2024, FLAP contributors'
author = 'S. Zoletnik, M. Vécsei, M. Vavrik, D. M. Takács'

language = 'en'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'numpydoc',
    'myst_parser',
    'sphinx_gallery.gen_gallery',
    'sphinx_copybutton',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# The master toctree document.
master_doc = 'index'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# These patterns also affects html_static_path and html_extra_path
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'furo'

html_theme_options = {
    "navigation_with_keys": True,
    "sidebar_hide_name": True,
    "source_repository": "https://github.com/fusion-flap/flap/",
    "source_branch": "master",
    "source_directory": "docs/",
}

html_logo = "static/FLAP_logo.png"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = [abspath('static')]

# -- Options for autosummary -------------------------------------------------

autoclass_content = 'both'
autosummary_generate = True

# -- Options for numpydoc ----------------------------------------------------

# All checks, except...
numpydoc_validation_checks = {'all', 'GL01', 'SS06', 'ES01', 'PR09', 'RT02', 'RT03', 'SA01', 'EX01'}

numpydoc_show_class_members = False
numpydoc_class_members_toctree = False


# -- Options for gen_gallery -------------------------------------------------

sphinx_gallery_conf = {
     'examples_dirs': '../../examples',  # path to your example scripts
     'gallery_dirs': 'auto_examples',  # path to where to save gallery generated output
}
