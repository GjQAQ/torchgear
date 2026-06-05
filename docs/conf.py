# Configuration file for the Sphinx documentation builder.

# -- Project information -----------------------------------------------------

project = 'torchgear'
copyright = '2026, Jiaqi Guo'
author = 'Jiaqi Guo'
release = '0.1.0'
version = release

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

autosummary_generate = True
autodoc_default_options = {
    'members': True,
    'undoc-members': False,
    'show-inheritance': True,
}
napoleon_google_docstring = False
napoleon_numpy_docstring = False

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
}

# -- Options for HTML output -------------------------------------------------

html_theme = 'furo'
html_title = 'torchgear'
html_static_path = ['_static']
html_baseurl = 'https://gjqaq.github.io/torchgear/'

# pygments_style = 'sphinx'
