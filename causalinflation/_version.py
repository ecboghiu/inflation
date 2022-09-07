import os

directory_of_this_file = os.path.dirname(os.path.abspath(__file__))

with open(f"{directory_of_this_file}/../VERSION.txt", "r") as f:
    __version__ = f.read().strip()