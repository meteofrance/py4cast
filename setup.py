from setuptools import find_packages, setup

# Build the long description from the README.md file
with open("README.md", "r") as readme:
    long_description = readme.read()


# Build the requirements list from the requirements.txt file
with open("requirements.txt") as requirements:
    required = requirements.read().splitlines()


# Instal the sources as a lib
setup(
    name="py4cast",
    # version="0.1.0", # Should py4cast have a version number ?
    author="PN-IA Météo France",
    description=(
        "Library to train a variety of Neural Network architectures "
        "on various weather forecasting datasets."
    ),
    long_description=long_description,
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=required,
)
