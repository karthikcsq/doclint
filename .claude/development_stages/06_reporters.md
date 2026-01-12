# 06 - Reporters

## Overview
Implement output formatters for console, JSON, and HTML reports.

## Subtasks

### 6.1 Base Reporter Interface
- [ ] Create `doclint/reporters/base.py`
- [ ] Define `BaseReporter` abstract class:
  - `name: str` class attribute
  - `report(results: Dict[str, List[Issue]], stats: Dict[str, Any]) -> None` abstract method
- [ ] Define common statistics structure:
  - `total_documents: int`
  - `total_issues: int`
  - `critical: int`
  - `warning: int`
  - `info: int`
  - `clean_documents: int`
  - `scan_duration: float`

### 6.2 Console Reporter - Header & Summary
- [ ] Create `doclint/reporters/console.py`
- [ ] Implement `ConsoleReporter` class extending `BaseReporter`
- [ ] Set `name = "console"`
- [ ] Initialize Rich Console in constructor
- [ ] Implement `_print_header()` method:
  - Use `console.rule()` for separator
  - Display scan summary stats
  - Use rich formatting
- [ ] Implement `_print_summary()` method:
  - Create Rich Table
  - Show total documents, issues by severity
  - Calculate and show clean document percentage
  - Use color coding (red for critical, yellow for warning, green for clean)

### 6.3 Console Reporter - Issue Display
- [ ] Implement `_print_issue()` method:
  - Create Rich Panel for each issue
  - Add severity icon (🔴 critical, ⚠️  warning, ℹ️ info)
  - Use color-coded borders
  - Display title, description, documents, details
  - Format details nicely
- [ ] Implement severity icon and color mappings:
  - CRITICAL: red, 🔴
  - WARNING: yellow, ⚠️
  - INFO: blue, ℹ️

### 6.4 Console Reporter - Main Report Method
- [ ] Implement `report()` method:
  - Print header
  - Group issues by severity
  - Print critical issues section
  - Print warnings section
  - Print info section
  - Print summary
  - Use rich formatting throughout
- [ ] Add option to suppress info/warning based on CLI flags

### 6.5 JSON Reporter
- [ ] Create `doclint/reporters/json.py`
- [ ] Implement `JSONReporter` class extending `BaseReporter`
- [ ] Set `name = "json"`
- [ ] Implement `report()` method:
  - Build JSON structure:
    - `scan_info` (timestamp, path, version, etc.)
    - `statistics` (counts by severity)
    - `issues` (array of issue dicts)
  - Convert Issues to dicts using `to_dict()`
  - Serialize with `json.dumps(indent=2)`
  - Print to stdout or write to file
- [ ] Add ISO 8601 timestamp
- [ ] Ensure all paths are strings (not Path objects)

### 6.6 JSON Reporter - File Output
- [ ] Add `output_path` parameter to reporter
- [ ] Implement file writing:
  - Write JSON to specified path
  - Create parent directories if needed
  - Handle write errors gracefully
- [ ] Add schema version to JSON output

### 6.7 HTML Reporter - Template
- [ ] Create `doclint/reporters/html.py`
- [ ] Implement `HTMLReporter` class extending `BaseReporter`
- [ ] Set `name = "html"`
- [ ] Create HTML template string:
  - Modern, responsive design
  - Use CSS for styling (embedded)
  - Color-coded severity badges
  - Expandable issue sections
  - Summary cards at top
  - No external dependencies

### 6.8 HTML Reporter - Report Generation
- [ ] Implement `report()` method:
  - Build HTML from template
  - Inject scan statistics
  - Generate issue HTML for each issue
  - Add syntax highlighting for code (if needed)
  - Write to file or return as string
- [ ] Implement `_render_issue()` helper:
  - Create HTML for single issue
  - Add collapsible details section
  - Format document paths as links (if possible)

### 6.9 HTML Reporter - Styling
- [ ] Add CSS for professional look:
  - Clean, modern design
  - Responsive layout
  - Color scheme matching severity
  - Print-friendly version
- [ ] Add JavaScript for interactivity:
  - Collapsible sections
  - Filter by severity
  - Search/filter functionality

### 6.10 Reporter Registry
- [ ] Create `doclint/reporters/registry.py`
- [ ] Implement `ReporterRegistry` class:
  - Store reporters in dictionary
  - `register(reporter: BaseReporter)` method
  - `get_reporter(name: str) -> BaseReporter` method
  - `get_available_reporters() -> List[str]` method
- [ ] Add default reporters registration

### 6.11 Reporter Module Initialization
- [ ] Update `doclint/reporters/__init__.py`:
  - Import all reporter classes
  - Export base and concrete reporters
  - Create default registry instance
  - Auto-register all reporters

### 6.12 Statistics Calculator
- [ ] Create `doclint/reporters/stats.py`
- [ ] Implement `calculate_stats()` function:
  - Accept results dict and document count
  - Calculate totals by severity
  - Calculate clean document count
  - Calculate scan duration
  - Return stats dict
- [ ] Add percentage calculations
- [ ] Add trend calculations (if previous scan available)

### 6.13 Unit Tests - Console Reporter
- [ ] Create `tests/test_reporters/test_console.py`
- [ ] Test header printing (capture console output)
- [ ] Test issue printing with different severities
- [ ] Test summary printing
- [ ] Test full report generation
- [ ] Test with no issues
- [ ] Test with many issues
- [ ] Mock Rich Console for testing

### 6.14 Unit Tests - JSON Reporter
- [ ] Create `tests/test_reporters/test_json.py`
- [ ] Test JSON structure
- [ ] Test JSON serialization
- [ ] Test file output
- [ ] Validate JSON schema
- [ ] Test with no issues
- [ ] Test path conversion to strings

### 6.15 Unit Tests - HTML Reporter
- [ ] Create `tests/test_reporters/test_html.py`
- [ ] Test HTML generation
- [ ] Test HTML validity (basic checks)
- [ ] Test file output
- [ ] Test with different issue types
- [ ] Test with no issues

### 6.16 Unit Tests - Reporter Registry
- [ ] Create `tests/test_reporters/test_registry.py`
- [ ] Test reporter registration
- [ ] Test reporter lookup
- [ ] Test listing available reporters

### 6.17 Integration Tests
- [ ] Create `tests/test_integration/test_reporting.py`
- [ ] Test full scan -> report pipeline
- [ ] Test switching between report formats
- [ ] Test report file generation
- [ ] Verify report contents match scan results

## Success Criteria
- ✅ All 3 reporters implemented (console, JSON, HTML)
- ✅ Console output is beautiful and readable
- ✅ JSON output is valid and machine-readable
- ✅ HTML output is professional and self-contained
- ✅ All unit tests passing (>85% coverage)
- ✅ Integration tests passing
- ✅ Reports accurately reflect scan results

## Dependencies
- Requires: Core Module (02), Detectors (05)
- Required by: CLI (08)

## Notes
- Console reporter should be the default
- Focus on clarity and actionability
- JSON format is critical for CI/CD integration
- HTML reports are great for sharing with non-technical stakeholders
- Use Rich library for beautiful terminal output
