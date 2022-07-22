This is the contribution guide for CausalInflation, a Python package for implementations of the inflation technique for causal inference. Contributions are very welcome and appreciated! 

You can begin contributing by opening an issue on the [GitHub repository](https://github.com/ecboghiu/inflation) to report bugs, request features, etc. Now we will outline some guidelines for contributing to the code, and for contributing to the documentation.

# How to contribute to the code

If you want to contribute to the code, please fork the repository and submit a pull request. 

## Install in develop mode

It is recommended to install the package in [develop mode](https://stackoverflow.com/questions/19048732/python-setup-py-develop-vs-install), which can be done with the following command in the terminal:
```
python setup.py develop
```
This way the package does not need to reinstalled after every modification.

## Testing
If you add new features, please write tests for them. Testing is done within the [Unit testing framework](https://docs.python.org/3/library/unittest.html). You can copy one of the existing tests to start with, and then adapt it to your code modification. To run the tests on your own computer, once you have cloned the repository, you can run the following command in the terminal:
```
python -m unittest -v
```
You can also run a specific test only, e.g. `python -m unittest -v test/test_sdp.py`. The `-v` option is used to print the test results.

If you are using a code editor, typically it includes a way to automatically find tests and run them. Here are specific instructions for some popular code editors:
- **Spyder**. Install the [unittest plugin](https://www.spyder-ide.org/blog/introducing-unittest-plugin/). Then create a new project in the `Projects` tab and choose as existing directory your local copy of the repository. Then in the `Run` tab there is the option of running tests.
- **Visual Studio Code**: Open universal search with `Ctrl+Shift+P` (universal search), search for `Configure Tests` and choose `unittest`. Then you can run all the tests in the `Test` tab in the Primary Side Bar.
- **PyCharm**: You can either create a new project and pull from the repository, or, if you have already downloaded the repository, you can open it in PyCharm. Then, under `File > Settings > Tools > Python Integrated Tools`, choose `Unittest` for `Default test runner`. Context actions are then updated and you can run individual tests ("play" arrow next to the test function) or all of them.

One suggestion is to write tests for a function in advance, and then write the function itself. This way you can easily see if your function is working as expected. Another suggestion is to you use breakpoints, which allow you to stop a program mid-way and inspect variable values, to debug your function. Most coding environments, such as [Visual Studio Code](https://code.visualstudio.com/docs/editor/debugging), support this feature.

###### Special note about test debugging for Visual Studio Code users

At the time of writing this (July 2022), there is a bug when trying to debug tests. It seems that the best [solution](https://github.com/microsoft/vscode-python/issues/10722) is to add the following to the `.vscode/launch.json` file:
```
{
    "name": "Python: Test debug config",
    "type": "python",
    "request": "test",
    "console": "integratedTerminal",
    "logToFile": true
}
```

## Style guidelines

CausalInflation code is developed according the standard practices in Python development. We do not have strict style guidelines, but we do have some suggestions. 
- In general, try to follow [PEP 8](https://peps.python.org/pep-0008/) (code) and [PEP 257](https://peps.python.org/pep-0257/) (docstrings) guidelines. 
- Avoid going over 80 characters per line.
- Write docstrings in [numpy style](https://numpydoc.readthedocs.io/en/latest/format.html), or else Sphinx automatic documentation generation will be buggy.
- Use annotations for [type hints](https://docs.python.org/3/library/typing.html) in the objects’ signature.


# How to contribute to the documentation

The documentation is built using Sphinx. The webpage is hosted with GitHub Pages. Currently, GitHup Pages is set to look for an `index.html` file in the `/docs` folder in the repository. Currently, this is not very clean, as small changes in the documentation lead to changing many HTML files, thus the commits are large. In the future, we will look into migrating the HTML build to the `gh-pages` branch of the repository, leading to a cleaner repository.

Since the documentation has already been created, you do not need to generate a configuration file. 

The main file is `docs/index.rst`, which outlines the structure of the documentation. One can update `index.rst` to include additional files or to remove files. It needs to be written in the [reStructuredText](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html) format.

###### Sphinx extensions

The following are the extensions added to Sphinx to generate the documentation, and you will need to have them installed if you want to locally build the documentation:

* `sphinx.ext.autodoc`: This extension is used to create `.rst` files from the class and method code docstrings.
* `sphinx.ext.napoleon`: This extension enables the correct identification of numpy-style docstrings.
* `sphinx.ext.githubpages`: This extension creates `.nojekyll` file on generated HTML directory to publish the document on GitHub Pages.
* `sphinx-rtd-theme`: for the Read the docs theme. 
* `nbsphinx`: to have Jupyter notebooks as pages.
* `sphinx.ext.autodoc.typehints`: to have typehints in the documentation.
* `m2r2`: this allows for the conversion of Markdown files to reStructuredText files.
  
These extensions and other specifications are found in the `conf.py` file. They allow for two important quality of life improvements in maintaining the documentation, namely, the ability to have Mardown documents and Jupyter notebooks as sections of the documentation. Future plans are to migrate everything to MyST serialisation.

**Note**: currently Sphinx expects all files to be placed under the source directors (`docs/`), so it seems difficult to include notebooks from the `examples/` folder into the documentation. The current workaround is to copy the notebooks to the `docs/` folder, however one needs to remember to update the other file when one of them is modified.

###### Update the API documentation 

If you want the API documentation to reflect the latest changes in the code, you can run the following command in the terminal in the `docs/` directory:

```sphinx-apidoc -f -o . ../causalinflation``` 

which will update the `.rst` files containing the module description. Note that after running this command, the title of the `modules.rst` will change back to `causalinflation`. Currently, the best solution seems to be to manually change it to `API Reference`.

###### Build the HTML documentation 


Once you have done all your modifications, you can build the documentation by running the following command in the terminal on the `docs/` directory:

```make html```

One can also build the documentation in other formats, such as PDF, using `make pdf`. To push to the webside your changes to the documentation, you can do so through a pull request.