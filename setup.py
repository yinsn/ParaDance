from pathlib import Path

from setuptools import find_packages, setup

SETUP_DIRECTORY = Path(__file__).resolve().parent

with (SETUP_DIRECTORY / "README.md").open() as ifs:
    LONG_DESCRIPTION = ifs.read()

install_requires = (
    [
        "pandas==2.0.1",
        "numpy==1.24.3",
        "matplotlib==3.7.1",
        "imageio==2.29.0",
        "tqdm==4.65.0",
        "scikit-learn==1.2.2",
        "matplotlib_venn==0.11.9",
    ],
)

setup(
    name="hiveviewer",
    version="0.0.1",
    author="Yin Cheng",
    author_email="yin.sjtu@gmail.com",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/yinsn/hiveviewer",
    python_requires=">=3.7",
    description="Offers a toolset for comprehensive, multi-faceted analysis of data exported from Hive, accompanied by powerful data visualization capabilities.",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=install_requires,
    include_package_data=True,
)
