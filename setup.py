from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="Ak-dskit",
    version="1.0.5",
    author="Aksh Agrawal",
    author_email="akshagr10@gmail.com",
    description="A comprehensive data science toolkit with 221+ functions for ML workflows",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Programmers-Paradise/DsKit",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.900",
        ],
        "full": [
            "plotly>=5.0.0",
            "wordcloud>=1.8.0",
            "nltk>=3.7",
            "textblob>=0.17.0",
            "hyperopt>=0.2.5",
            "optuna>=2.10.0",
            "xgboost>=1.5.0",
            "lightgbm>=3.3.0",
            "catboost>=1.0.0",
            "imbalanced-learn>=0.8.0",
            "pandas-profiling>=3.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "dskit=dskit.cli:main",
        ],
    },
    keywords="data science, machine learning, eda, preprocessing, automl, feature engineering",
    project_urls={
        "Bug Reports": "https://github.com/Programmers-Paradise/DsKit/issues",
        "Source": "https://github.com/Programmers-Paradise/DsKit",
        "Documentation": "https://github.com/Programmers-Paradise/DsKit/blob/main/COMPLETE_FEATURE_DOCUMENTATION.md",
        "Changelog": "https://github.com/Programmers-Paradise/DsKit/releases",
    },
)