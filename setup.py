from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setup(
    name="monotonicnetworks",
    version="1.5.2",
    packages=["monotonicnetworks"],
    license="MIT",
    author="Ouail Kitouni, Niklas Nolte",
    author_email="kitouni@mit.edu, nolte@meta.com",
    description=(
        "Pytorch implementation of constrained weight operator norms for robustness "
        + "and monotonicity in neural networks."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/niklasnolte/MonotonicNetworks",
)
