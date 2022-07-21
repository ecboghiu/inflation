# Contributing to CausalInflation

This is the contribution guide for CausalInflation, a Python package for implementations of the inflation technique for causal inference. Contributions are very welcome and appreciated! 

You can begin contributing by opening an issue on the [GitHub repository](https://github.com/ecboghiu/inflation) to report bugs, request features, etc. If you want to contribute to the code, please fork the repository and submit a pull request. 

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


## Updating the documentation

The [documentation](https://ecboghiu.github.io/inflation/_build/html/index.html) is built using Sphinx. The webpage is hosted with GitHub Pages. To update the documentation, you can do so through a pull request. More details on how to build the documentation can be found on the [documentation contribution page](https://ecboghiu.github.io/inflation/_build/html/contribute.html).