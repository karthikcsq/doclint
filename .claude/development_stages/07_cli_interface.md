# 07 - CLI Interface

## Overview
Implement the command-line interface using Typer and Rich for beautiful terminal interaction.

## Subtasks

### 7.1 Main CLI App Setup
- [ ] Create `doclint/cli/main.py`
- [ ] Initialize Typer app:
  ```python
  app = typer.Typer(
      name="doclint",
      help="Data quality linting for AI knowledge bases",
      add_completion=False,
  )
  ```
- [ ] Add version callback
- [ ] Add rich help formatting
- [ ] Configure command structure

### 7.2 Scan Command - Basic Structure
- [ ] Create `doclint/cli/scan.py`
- [ ] Define `scan()` command function:
  - Path argument (directory to scan)
  - --recursive / --no-recursive flag
  - --config option for config file
  - --format option (console/json/html)
  - --output option for file output
  - --detectors option for selective detection
  - --severity option to filter by severity
  - --cache-dir option
  - --no-cache flag
  - --workers option
  - --verbose / --quiet flags
- [ ] Add rich help text and examples

### 7.3 Scan Command - Argument Validation
- [ ] Validate path exists and is directory
- [ ] Validate config file path if provided
- [ ] Validate format is valid choice
- [ ] Validate severity is valid choice
- [ ] Validate detector names are valid
- [ ] Validate workers count > 0
- [ ] Provide helpful error messages

### 7.4 Scan Command - Configuration Loading
- [ ] Load configuration in priority order:
  1. CLI arguments (highest)
  2. --config file
  3. .doclint.toml in scan directory
  4. ~/.config/doclint/config.toml
  5. Default config
- [ ] Merge configuration sources
- [ ] Override config with CLI arguments

### 7.5 Scan Command - Component Initialization
- [ ] Initialize ParserRegistry
- [ ] Initialize EmbeddingGenerator (with config)
- [ ] Initialize CacheManager (or disable if --no-cache)
- [ ] Initialize DetectorRegistry
- [ ] Initialize ReporterRegistry
- [ ] Create Scanner instance
- [ ] Handle initialization errors gracefully

### 7.6 Scan Command - Progress Display
- [ ] Create Rich Progress instance
- [ ] Add progress tasks:
  - File discovery
  - Document parsing
  - Embedding generation
  - Running detectors
- [ ] Update progress in real-time
- [ ] Show spinner for long operations
- [ ] Display current file being processed

### 7.7 Scan Command - Execution
- [ ] Run async scanner.scan_directory()
- [ ] Handle errors during scan
- [ ] Show progress updates
- [ ] Collect results
- [ ] Calculate statistics
- [ ] Generate report using selected reporter
- [ ] Output to console or file

### 7.8 Scan Command - Exit Codes
- [ ] Return 0 if no issues found
- [ ] Return 1 if any issues found
- [ ] Return 2 if critical issues found
- [ ] Return 3 if scan error occurred
- [ ] Document exit codes in help text

### 7.9 Version Command
- [ ] Create `doclint/cli/version.py`
- [ ] Implement `version()` command:
  - Display DocLint version
  - Display Python version
  - Display platform information
  - Display installed dependencies versions
- [ ] Add to main app

### 7.10 Config Command - Init
- [ ] Create `doclint/cli/config.py`
- [ ] Implement `config init` command:
  - Create default config file
  - Prompt for location (.doclint.toml or ~/.config/doclint/)
  - Include all default settings with comments
  - Confirm creation

### 7.11 Config Command - Show
- [ ] Implement `config show` command:
  - Load current configuration
  - Display all settings
  - Show source of each setting (file, default)
  - Format nicely with Rich

### 7.12 Config Command - Validate
- [ ] Implement `config validate` command:
  - Load config file
  - Validate with Pydantic
  - Report validation errors
  - Report success if valid

### 7.13 Cache Command - Stats
- [ ] Create `doclint/cli/cache.py`
- [ ] Implement `cache stats` command:
  - Load CacheManager
  - Get cache statistics
  - Display cache size (MB)
  - Display item count
  - Display cache location
  - Display hit rate if available

### 7.14 Cache Command - Clear
- [ ] Implement `cache clear` command:
  - Load CacheManager
  - Show current cache size
  - Confirm before clearing (unless --force)
  - Clear cache
  - Report success

### 7.15 Global Options
- [ ] Add global --verbose flag
- [ ] Add global --quiet flag
- [ ] Add global --no-color flag
- [ ] Configure logging based on verbosity
- [ ] Set up Rich Console with color settings

### 7.16 Error Handling
- [ ] Create `doclint/cli/errors.py`
- [ ] Define CLI exception classes
- [ ] Implement global exception handler
- [ ] Format errors nicely with Rich
- [ ] Add suggestions for common errors
- [ ] Log errors to file if verbose

### 7.17 Help Text and Examples
- [ ] Write comprehensive help text for each command
- [ ] Add usage examples to help
- [ ] Add links to documentation
- [ ] Format help with Rich panels

### 7.18 Autocomplete (Optional)
- [ ] Enable shell completion for Typer
- [ ] Test with bash, zsh
- [ ] Document installation of completions

### 7.19 Unit Tests - Scan Command
- [ ] Create `tests/test_cli/test_scan.py`
- [ ] Test argument parsing
- [ ] Test path validation
- [ ] Test configuration loading
- [ ] Test exit codes
- [ ] Test error handling
- [ ] Use CliRunner for testing

### 7.20 Unit Tests - Other Commands
- [ ] Create `tests/test_cli/test_version.py`
- [ ] Create `tests/test_cli/test_config.py`
- [ ] Create `tests/test_cli/test_cache.py`
- [ ] Test each command's functionality
- [ ] Test help text generation

### 7.21 E2E Tests
- [ ] Create `tests/test_e2e/test_cli_scan.py`
- [ ] Test full scan workflow from CLI
- [ ] Test with real files
- [ ] Test different output formats
- [ ] Test error scenarios
- [ ] Verify exit codes

### 7.22 CLI Documentation
- [ ] Create usage examples in docs/
- [ ] Document all commands and options
- [ ] Add common workflows
- [ ] Add troubleshooting guide

## Success Criteria
- ✅ All commands implemented (scan, version, config, cache)
- ✅ Beautiful terminal output with Rich
- ✅ Proper argument validation and error handling
- ✅ Exit codes working correctly
- ✅ All CLI tests passing
- ✅ E2E tests passing
- ✅ Help text is comprehensive and helpful

## Dependencies
- Requires: All previous components (01-06)
- Required by: E2E tests, documentation

## Notes
- The CLI is the main user interface - it must be polished
- Focus on great UX with helpful messages
- Use Rich for beautiful output
- Test with various input scenarios
- Consider edge cases (empty directory, no permissions, etc.)
