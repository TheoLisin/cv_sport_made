import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="CV sport classification task",
    author="Theo",
    author_email="theo.lisin@gmail.com",
    description="Python project",
    keywords="Python, CV, classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    version="0.1.0",
    classifiers=[
        # see https://pypi.org/classifiers/
        "Development Status :: 1 - Alpha",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "kaggle",
        "jupyterlab",
        "lightning",
        "tqdm",
        "numpy",
        "matplotlib",
        "pandas",
        "opencv-python",
        "scikit-learn",
    ],
    extras_require={
        "dev": [
            "jupyterlab",
            "torchsummary",
            "ipywidgets",
            "wemake-python-styleguide",
            "mypy",
            "black",
        ]
    }
)