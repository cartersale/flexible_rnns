from setuptools import find_packages, setup


setup(
    name="flexsym",
    version="0.1.0",
    description="Connectivity symmetry and dynamical motifs in multitask RNNs",
    package_dir={"": "src"},
    packages=find_packages("src"),
)