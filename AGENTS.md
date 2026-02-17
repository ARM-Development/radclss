# Agent Instructions

## Git Commit Protocol

Before creating ANY git commit, you MUST:

1. **Run unit tests** to ensure code changes don't break existing functionality:
   ```bash
   pytest tests/
   ```

2. **Run pre-commit hooks** on all staged files:
   ```bash
   pre-commit run --all-files
   ```

3. **If tests fail:**
   - Review the test failures and error messages
   - Fix the code to make tests pass
   - Re-run tests to verify all tests pass
   - DO NOT proceed with commit until all tests pass

4. **If pre-commit fails:**
   - Review the linting errors and warnings
   - Fix all issues automatically where possible (pre-commit often auto-fixes)
   - Stage the auto-fixed files
   - Re-run pre-commit to verify all issues are resolved
   - Repeat until pre-commit passes successfully

5. **Only after BOTH tests and pre-commit pass**, proceed with the git commit

6. **NEVER:**
   - Commit without running tests first
   - Commit without running pre-commit first
   - Use `--no-verify` to skip pre-commit hooks
   - Ignore test failures, linting errors, or warnings
   - Skip tests with `-k` or `-m` flags to avoid failing tests

## Pre-commit Tools Used

This project uses the following linting and formatting tools via pre-commit:
- **black**: Python code formatter
- **ruff**: Fast Python linter (with auto-fix enabled)
- **pyupgrade**: Automatically upgrades Python syntax for newer versions (Python 3.10+)
- **pre-commit-hooks**: trailing whitespace, end-of-file fixer, YAML/JSON validation, etc.

## Unit Testing Requirements

When making code changes, you MUST:

1. **Run existing tests** to ensure no regressions:
   ```bash
   pytest tests/
   ```

2. **Write new tests** for new functionality:
   - Add test functions to appropriate test files in `tests/` directory
   - Follow existing test patterns (see `tests/test_*.py` files)
   - Test edge cases, error conditions, and expected behavior
   - Ensure new tests pass before committing

3. **Update tests** when modifying existing functionality:
   - Update affected test cases to reflect new behavior
   - Ensure all tests still pass after changes

4. **Test file organization:**
   - `tests/test_config.py` - Configuration testing
   - `tests/test_io.py` - I/O functionality testing
   - `tests/test_radclss.py` - Core radclss functionality
   - `tests/test_vis.py` - Visualization testing

## Example Workflow

```bash
# 1. Make your code changes
# ... edit files ...

# 2. Run tests to verify nothing breaks
pytest tests/

# 3. If you added new functionality, write tests for it
# ... create/update test files ...

# 4. Run tests again to ensure new tests pass
pytest tests/

# 5. Stage your changes
git add file1.py file2.py tests/test_new_feature.py

# 6. Run pre-commit
pre-commit run --all-files

# 7. If pre-commit made fixes, stage them
git add -u

# 8. Re-run to verify
pre-commit run --all-files

# 9. Once tests and pre-commit pass, commit
git commit -m "Your commit message"
```

## Installation

If pytest or pre-commit is not installed, install them first:
```bash
# Install pytest for running tests
pip install pytest

# Install pre-commit for linting hooks
pip install pre-commit
pre-commit install
```

## Running Specific Tests

```bash
# Run all tests
pytest tests/

# Run a specific test file
pytest tests/test_config.py

# Run a specific test function
pytest tests/test_config.py::test_function_name

# Run tests with verbose output
pytest tests/ -v

# Run tests and show print statements
pytest tests/ -s
```
