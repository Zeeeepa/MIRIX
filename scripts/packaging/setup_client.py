#!/usr/bin/env python3
"""
Mirix Client Package Setup
===========================

This setup script packages ONLY the client-side components of Mirix.
The client package is lightweight and contains only what's needed to
communicate with a Mirix server.

Package Name: mirix-client
Purpose: Remote client library for Mirix server
"""

import os

from setuptools import find_packages, setup

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
# Go up two levels: packaging/ -> scripts/ -> project_root/
project_root = os.path.dirname(os.path.dirname(this_directory))

with open(os.path.join(project_root, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


# Get version
def get_version():
    import re

    version_file = os.path.join(project_root, "mirix", "__init__.py")
    with open(version_file, "r", encoding="utf-8") as version_f:
        version_match = re.search(
            r"^__version__ = ['\"]([^'\"]*)['\"]", version_f.read(), re.M
        )
        if version_match:
            return version_match.group(1)
        raise RuntimeError("Unable to find version string.")


# Client-specific dependencies (minimal)
client_dependencies = [
    "requests>=2.31.0",
    "pydantic>=2.0.0",
    "pydantic-settings>=2.0.0",
    "python-dotenv>=1.0.0",
    "httpx>=0.25.0",
    "jinja2>=3.1.0",
    "demjson3>=3.0.0",
    "json-repair>=0.25.0",
    "rich>=13.7.1,<14.0.0",
    "pytz>=2024.1",
    "docstring-parser>=0.15",
    "pyhumps>=3.8.0",
]

# Change to project root directory so we can use relative paths
os.chdir(project_root)

setup(
    name="jl-ecms-client",
    version=get_version(),
    author="Mirix AI",
    author_email="yuwang@mirix.io",
    description="Mirix Client - Lightweight Python client for Mirix server",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Mirix-AI/MIRIX",
    project_urls={
        "Documentation": "https://docs.mirix.io",
        "Website": "https://mirix.io",
        "Source Code": "https://github.com/Mirix-AI/MIRIX",
        "Bug Reports": "https://github.com/Mirix-AI/MIRIX/issues",
    },
    # Only include client-related packages
    packages=[
        "mirix.client",
        "mirix.schemas",
        "mirix.schemas.openai",
        "mirix.helpers",
        "mirix.functions",
        "mirix.functions.function_sets",
    ],
    py_modules=[
        "mirix.system",
        "mirix.settings",
        "mirix.log",
        "mirix.constants",
        "mirix.errors",
    ],
    package_dir={"": "."},
    include_package_data=True,
    package_data={
        "mirix": [],
    },
    install_requires=client_dependencies,
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-asyncio",
            "black",
            "isort",
            "flake8",
            "mypy",
        ],
    },
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Communications :: Chat",
    ],
    keywords="ai, memory, agent, llm, assistant, client, api",
    license="Apache License 2.0",
    zip_safe=False,
)
