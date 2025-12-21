"""
Setup configuration for Chart2CSV.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    requirements = [
        line.strip()
        for line in requirements_path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
else:
    requirements = []

setup(
    name="chart2csv",
    version="0.1.0",
    author="KikuAI Lab",
    author_email="contact@kikuai.dev",
    description="Extract data from chart images to CSV/JSON with honest confidence scoring",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/KikuAI-Lab/Chart2CSV",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: Visualization",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "mypy>=1.5.0",
            "ruff>=0.0.280",
        ],
        "pdf": [
            "pypdfium2>=4.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "chart2csv=chart2csv.cli.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "chart2csv": ["py.typed"],
    },
)
