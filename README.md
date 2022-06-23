# UnifiedInflation
Implementations of the Inflation Technique for Causal Inference. The complete documentation is hosted online [here](https://ecboghiu.github.io/inflation/).

## Installation

To install the package, run the following command:

```
pip install causalinflation
```

You can also install the latest developed version with  `pip install git+https://github.com/ecboghiu/inflation.git@main` or clone the repository and run `python setup.py install`. If you forsee making many contributions, you can install the package in develop mode: `python setup.py develop`.




## Temporary notes on how to contribute 

### Documentation

Documentation is built using Sphinx. Here are quick notes on how some sphinx commands work. 

The documentation is already built. One can update index.rst to include additional files. The sphinx configuration file `conf.py` has been modified to allow including files from the examples folder.

To modify the current files, one can modify the `.rst` files already included in `index.rst`, for example, `tutorial.rst`. To include the `.rst` files detailing the code description, one should run `sphinx-apidoc -f -o . ../causalinflation` in the `./docs`. This will update the corresponding `.rst` files detailing the code. To build the documentation again with the updated files, one can run `make html` and then push to update the webpage.

Note that we need the following extensions with Sphinx, some might need to be installed locally on your machine:
 - `sphinx.ext.autodoc`: This extension is used to create `.rst` files from the class and method code docstrings.
 - `sphinx.ext.napoleon`: This extension enables the correct identification of numpy-style docstrings.
 - `sphinx.ext.githubpages`: This extension creates `.nojekyll` file on generated HTML directory to publish the document on GitHub Pages. 
  - `sphinx-rtd-theme`: for the Read the docs theme.
  - `nbsphinx`: to have Jupyter notebooks as pages.

### Short note on tests:
- If using Spyder: install the unittest plugin. See here https://www.spyder-ide.org/blog/introducing-unittest-plugin/ Then create a new project in the "Projects" tab and choose as existing directory your local repo copy. Then in the "Run" tab there is the option of running tests.
- If using VSCode: Ctrl+Shift+P (universal search) "Configure Tests" and choose unittest. Then you run all the tests in the "Test" tab on the left.
- If using PyCharm: Either make a new project and pull from the repository, or open the already downloaded one. Then in File > Settings > Tools > Python Integrated Tools choose "Default test runner" as "Unittest". Then context actions are updated and you can run individual tests ("play" arrow next to the test function) or all of them.
- If using terminal: be in the UnifiedInflation folder (and not in test/). Run "python -m unittest -v" to run all the tests ("-v" for verbose), or "python -m unittest -v test/test_hypergraph.py" to run a specific test script