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
CLIENT_VERSION = "0.1.0"

with open(os.path.join(project_root, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


def load_requirements(filename: str) -> list[str]:
    """Load dependency list from a requirements file, ignoring comments/blanks."""
    req_path = os.path.join(this_directory, filename)
    with open(req_path, encoding="utf-8") as req_file:
        return [
            line.strip()
            for line in req_file.readlines()
            if line.strip() and not line.startswith("#")
        ]


# Client-specific dependencies (minimal)
client_dependencies = load_requirements("requirements_client.txt")

# Change to project root directory so we can use relative paths
os.chdir(project_root)

setup(
    name="mirix-client",
    version=CLIENT_VERSION,
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
    # Only include client-related packages (explicit to avoid pulling server code)
    packages=[
        "mirix.client",
        "mirix.schemas",
        "mirix.schemas.openai",
        "mirix.helpers",
    ],
    py_modules=[
        "mirix.system",
        "mirix.settings",
        "mirix.log",
        "mirix.constants",
    ],
    package_dir={"": "."},
    include_package_data=False,
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
