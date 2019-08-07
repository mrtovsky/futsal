import codecs
import json
import pathlib
import re

from setuptools import find_packages, setup


PACKAGE_PATH = pathlib.Path(__file__).parent


def read(*paths):
    file = str(PACKAGE_PATH.joinpath(*paths).resolve())
    with codecs.open(file) as f:
        return f.read()


def find_version(*paths):
    version_file = read(*paths)
    version_pattern = re.compile(r"^__version__ = ['\"]([^'\"]*)['\"]", re.M)
    version_match = version_pattern.search(version_file)
    if version_match:
        return version_match.group(1)
    else:
        raise RuntimeError('Unable to find version string.')


with codecs.open("requirements.txt") as f:
    REQUIREMENTS = f.read().splitlines()

with codecs.open("extras_requirements.json", "r") as f:
    EXTRAS_REQUIREMENTS = json.load(f)

DESCRIPTION = "Forecasting the future level of sales for a certain " \
              "chain of stores."
LONG_DESCRIPTION = read("README.md")

setup(
    name="futsal",
    version=find_version("futsal", "__init__.py"),
    author="Mateusz Zakrzewski",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    install_requires=REQUIREMENTS,
    extras_require=EXTRAS_REQUIREMENTS,
    packages=find_packages("."),
    zip_safe=True
)
