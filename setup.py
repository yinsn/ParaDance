from pathlib import Path

from setuptools import find_packages, setup

SETUP_DIRECTORY = Path(__file__).resolve().parent

with (SETUP_DIRECTORY / "README.md").open() as ifs:
    LONG_DESCRIPTION = ifs.read()

install_requires = (
    [
        "imageio>=2.31.1",
        "joblib>=1.2.0",
        "matplotlib>=3.7.1",
        "matplotlib_venn>=0.11.10",
        "numpy>=1.20.3",
        "openpyxl>=3.1.2",
        "optuna>=3.1.1",
        "pandas>=2.0.1",
        "scikit-learn>=1.2.2",
        "scipy>=1.11.3",
        "sqlalchemy>=2.0.15",
        "tqdm>=4.65.0",
    ],
)

__version__ = "0.3.17"

setup(
    name="paradance",
    version=__version__,
    author="Yin Cheng",
    author_email="yin.sjtu@gmail.com",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/yinsn/paradance",
    python_requires=">=3.6",
    description="Offers a toolset for comprehensive, multi-faceted analysis of data exported from Hive, accompanied by powerful data visualization capabilities.",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=install_requires,
    include_package_data=True,
)
