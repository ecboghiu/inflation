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
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------

project = "CausalInflation"
author = "Emanuel-Cristian Boghiu, Elie Wolfe, Alejandro Pozas-Kerstjens"
copyright = "2022, " + author

# The full version, including alpha/beta/rc tags
release = "0.1"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ["m2r2",
              "nbsphinx",
              "sphinx.ext.autodoc",
              "sphinx.ext.autodoc.typehints",
              "sphinx.ext.githubpages",
              "sphinx.ext.napoleon",
              "sphinx.ext.viewcode",
              "sphinx_copybutton"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "../test*", "test*"]

# Specify the file where the table of contents is, so it appears only on the
# sidebar. From
# https://stackoverflow.com/questions/54348962/sphinx-toctree-in-sidebar-only

master_doc = "contents"

# Make that the index page does not disappear from sidebar TOC. From
# https://stackoverflow.com/questions/18969093/how-to-include-the-toctree-in-the-sidebar-of-each-page

html_sidebars = {"**": ["globaltoc.html", "relations.html",
                        "sourcelink.html", "searchbox.html"]}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"  # "alabaster"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Remove bash indicators from copied text
copybutton_prompt_text = "$ "
