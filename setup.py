import os
from setuptools import setup

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "causalinflation",
    version = "0.1",
    author = "Emanuel-Cristian Boghiu, Alejandro Pozas-Kerstjens",
    author_email = "cristian.boghiu@icfo.eu, physics@alexpozas.com",
    description = ("Implementations of the Inflation Technique for Causal Inference"),
    license = "Creative Commons License",
    url = "https://github.com/ecboghiu/inflation",
    packages=['causalinflation'] #long_description=read('README.md')
)