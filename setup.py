from pathlib import Path

from setuptools import find_packages, setup

SETUP_DIRECTORY = Path(__file__).resolve().parent

with (SETUP_DIRECTORY / "README.md").open() as ifs:
    LONG_DESCRIPTION = ifs.read()

install_requires = (
    [
        "pandas",
        "numpy",
        "matplotlib",
        "imageio",
        "tqdm",
        "scikit-learn",
        "matplotlib_venn",
        "openpyxl",
        "optuna",
    ],
)

setup(
    name="hiveviewer",
    version="0.2.5",
    author="Yin Cheng",
    author_email="yin.sjtu@gmail.com",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/yinsn/hiveviewer",
    python_requires=">=3.6",
    description="Offers a toolset for comprehensive, multi-faceted analysis of data exported from Hive, accompanied by powerful data visualization capabilities.",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=install_requires,
    include_package_data=True,
)
