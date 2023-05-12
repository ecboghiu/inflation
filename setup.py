from setuptools import setup, find_packages

with open('inflation/_version.py') as f:
    exec(f.read())

setup(
    name="inflation",
    version=__version__,
    install_requires=["numpy", "sympy", "scipy", "numba", "mosek", "tqdm"],
    extras_require={
        "docs": ["nbsphinx", "m2r2", "sphinx_rtd_theme", "sphinx_copybutton"]
    },
    author="Emanuel-Cristian Boghiu, Elie Wolfe, Alejandro Pozas-Kerstjens",
    author_email="cristian.boghiu@icfo.eu, ewolfe@pitp.ca, physics@alexpozas.com",
    description="Implementations of the Inflation Technique for Causal Inference",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
    packages=find_packages(exclude=["test", "doc*", "example*"]),
    license="GNU GPL v. 3.0",
    url="https://github.com/ecboghiu/inflation",
    project_urls={"Documentation": "https://ecboghiu.github.io/inflation/_build/html/index.html",
                  "Source": "https://github.com/ecboghiu/inflation",
                  "Issue Tracker": "https://github.com/ecboghiu/inflation/issues"},
    zip_safe=False,  # To avoid problems with Numba, https://github.com/numba/numba/issues/4908
)
