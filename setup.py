from setuptools import setup, find_packages
import os

version_dict = {}
version_file = os.path.join(os.path.dirname(__file__), "pypipr", "_version.py")
with open(version_file, "r", encoding="utf-8") as f:
    exec(f.read(), version_dict) # type: ignore

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="pypipr",
    version=version_dict["__version__"], # type: ignore
    author="S. Belgers",
    author_email="s.belgers@tue.nl",
    description="Package for processing pupil data, with a focus on the post illumination pupil response.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SBelgers/pypipr_package",
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    package_data={"": ["examples/*.ipynb"], "pypipr": ["data/*"]},
    python_requires=">=3.12",
)
