from setuptools import setup
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="monotonicnetworks",
    version="1.5.1",
    packages=["monotonicnetworks"],
    url="",
    license="MIT",
    author="Ouail Kitouni, Niklas Nolte",
    author_email="kitouni@mit.edu, nolte@meta.com",
    description=(
        "Pytorch implementation of constrained weight operator norms for robustness "
        + "and monotonicity in neural networks."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
)
