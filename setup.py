from setuptools import setup, find_packages

with open("requirements.txt") as f:
    install_requires = f.read().splitlines()

setup(
    name="deep-mt",
    version="0.1.0",
    description="Prediction of outcome of Mechanical Thrombectomy in endovascular stroke patients",
    license="Apache License 2.0",
    author="James Diprose, Tuan Chien",
    author_email="jamie.diprose@gmail.com",
    url="https://github.com/jdddog/deep-mt",
    packages=find_packages(),
    install_requires=install_requires,
    test_suite="tests",
    entry_points={"console_scripts": ["deep-mt = deep_mt.cli:cli"]},
)
