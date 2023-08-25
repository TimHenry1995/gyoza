# Configuration file for the Sphinx documentation builder.
#

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "private-members": True
}

# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os, sys
for x in os.walk('.'):
  sys.path.insert(0, x[0])

for x in os.walk('..'):
  sys.path.insert(0, x[0])

for x in os.walk(os.path.join("..","..")):
  sys.path.insert(0, x[0])


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'gyoza'
copyright = '2023, Tim Dick'
author = 'Tim Dick'
release = '0.1.2'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.todo','sphinx.ext.viewcode','sphinx.ext.autodoc']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['']
