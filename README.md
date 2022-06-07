# UnifiedInflation
Implementations of the Inflation Technique for Causal Inference

Short note on tests:
- If using Spyder: install the unittest plugin. See here https://www.spyder-ide.org/blog/introducing-unittest-plugin/ Then create a new project in the "Projects" tab and choose as existing directory your local repo copy. Then in the "Run" tab there is the option of running tests.
- If using VSCode: Ctrl+Shift+P (universal search) "Configure Tests" and choose unittest. Then you run all the tests in the "Test" tab on the left.
- If using PyCharm: Either make a new project and pull from the repository, or open the already downloaded one. Then in File > Settings > Tools > Python Integrated Tools choose "Default test runner" as "Unittest". Then context actions are updated and you can run individual tests ("play" arrow next to the test function) or all of them.
- If using terminal: be in the UnifiedInflation folder (and not in test/). Run "python -m unittest -v" to run all the tests ("-v" for verbose), or "python -m unittest -v test/test_hypergraph.py" to run a specific test script