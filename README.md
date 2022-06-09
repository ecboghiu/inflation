# UnifiedInflation
Implementations of the Inflation Technique for Causal Inference

Short note on tests:
- If using Spyder: install the unittest plugin. See here https://www.spyder-ide.org/blog/introducing-unittest-plugin/ Then create a new project in the "Projects" tab and choose as existing directory your local repo copy. Then in the "Run" tab there is the option of running tests.
- If using VSCode: Ctrl+Shift+P (universal search) "Configure Tests" and choose unittest. Then you run all the tests in the "Test" tab on the left.
- If using PyCharm: Either make a new project and pull from the repository, or open the already downloaded one. Then in File > Settings > Tools > Python Integrated Tools choose "Default test runner" as "Unittest". Then context actions are updated and you can run individual tests ("play" arrow next to the test function) or all of them.
- If using terminal: be in the UnifiedInflation folder (and not in test/). Run "python -m unittest -v" to run all the tests ("-v" for verbose), or "python -m unittest -v test/test_hypergraph.py" to run a specific test script

## On contributing to the documentation

Documentation is built using Sphinx. Here are quick notes on how some sphinx commands work. 

The documentation is already built. One can update index.rst to include additional files. To modify the current files, one can modify the .rst files already included in index.rst, for example, `tutorial.rst`. To include the .rst files detailing the code description, one should run `sphinx-apidoc -f -o . ..` in the `./docs` folder. This will update the corresponding .rst files detailing the code. To build the documentation again with the updated files, one can run `make html`.

Note that we need the following extensions with Sphinx, which you should have installed locally:
 - sphinx.ext.autodoc: This extension is used to create .rst files from the class and method code docstrings.
 - sphinx.ext.napoleon: This extension enables the correct identification of numpy-style docstrings.
 - sphinx.ext.githubpages: This extension creates .nojekyll file on generated HTML directory to publish the document on GitHub Pages. 

 One needs the following Sphinx extensions in order to locally compile:
  - sphinx-rtd-theme: for the Read the docs theme
  - nbsphinx: to have Jupyter notebooks as pages