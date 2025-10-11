# MIRIX Code Standards

## Follow PEP8 style guide

[PEP 8](https://peps.python.org/pep-0008/) is the official style guide for Python code, promoting readability and consistency. Adhering to it improves code maintainability and collaboration. Key aspects include:

- **Indentation**: Use 4 spaces per indentation level. Avoid using tabs.
- **Line Length**: Limit lines to a maximum of 79 characters for code, and 72 characters for docstrings and comments.
- **Blank Lines**: Use two blank lines to separate top-level functions and classes. Use one blank line to separate methods within a class. 
- **Imports**: Place all import statements at the top of the file, after any module comments and docstrings. Group imports by type: standard library, third-party, and local application imports, with a blank line between each group.
- **Naming Conventions**:
  - **Variables and Functions**: Use snake_case (lowercase with underscores).
  - **Classes**: Use CamelCase (each word capitalized).
  - **Constants**: Use ALL_CAPS_WITH_UNDERSCORES.
- **Whitespace**: Use spaces around operators and after commas, but not directly inside parentheses or brackets. 
- **Comments**: Use inline comments sparingly and ensure they add value by explaining why a piece of code exists, not just what it does. Separate inline comments from the statement by at least two spaces. 
- **Docstrings**: Use triple double quotes for docstrings and ensure they clearly describe the purpose of modules, classes, and functions.

Tools like linters (e.g., [pycodestyle](https://pypi.org/project/pycodestyle/)) and auto-formatters (e.g., [autopep8](https://pypi.org/project/autopep8/)) can assist in ensuring PEP 8 compliance. Many modern IDEs also offer integrated PEP 8 checking and formatting features. In MIRIX, we use [Ruff](https://docs.astral.sh/ruff/) and [Pyright](https://microsoft.github.io/pyright/#/). Both tools are included in the pyproject.toml and requirements.txt