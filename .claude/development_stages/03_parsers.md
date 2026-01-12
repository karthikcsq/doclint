# 03 - Document Parsers

## Overview
Implement parsers for different document formats (PDF, DOCX, HTML, Markdown, Text).

## Subtasks

### 3.1 Base Parser Interface
- [ ] Create `doclint/parsers/base.py`
- [ ] Define `BaseParser` abstract class:
  - `file_type: str` class attribute
  - `supported_extensions: list[str]` class attribute
  - `parse(path: Path) -> str` abstract method
  - `extract_metadata(path: Path) -> DocumentMetadata` abstract method
  - `can_parse(path: Path) -> bool` method
  - `clean_text(text: str) -> str` static method
- [ ] Implement `clean_text()`:
  - Remove excessive whitespace
  - Remove null bytes
  - Strip and normalize text
- [ ] Add comprehensive docstrings

### 3.2 PDF Parser Implementation
- [ ] Create `doclint/parsers/pdf.py`
- [ ] Implement `PDFParser` class extending `BaseParser`
- [ ] Set `file_type = "pdf"` and `supported_extensions = ['.pdf']`
- [ ] Implement `parse()` method:
  - Open PDF with `pypdf.PdfReader`
  - Extract text from all pages
  - Join page text with double newlines
  - Clean and return text
  - Handle errors gracefully
- [ ] Implement `extract_metadata()` method:
  - Read PDF metadata (author, title)
  - Parse creation date (PDF format: D:YYYYMMDDHHmmSS)
  - Parse modification date
  - Return DocumentMetadata object
- [ ] Implement `_parse_pdf_date()` helper:
  - Parse PDF date format to datetime
  - Handle various date formats
  - Return None on parse failure

### 3.3 Markdown Parser Implementation
- [ ] Create `doclint/parsers/markdown.py`
- [ ] Implement `MarkdownParser` class extending `BaseParser`
- [ ] Set `file_type = "markdown"` and `supported_extensions = ['.md', '.markdown']`
- [ ] Implement `parse()` method:
  - Read raw markdown file
  - Optionally convert to plain text or keep markdown
  - Clean and return text
- [ ] Implement `extract_metadata()` method:
  - Parse YAML frontmatter if present
  - Extract title from first H1 heading
  - Get file system dates
  - Return DocumentMetadata

### 3.4 Plain Text Parser Implementation
- [ ] Create `doclint/parsers/text.py`
- [ ] Implement `TextParser` class extending `BaseParser`
- [ ] Set `file_type = "text"` and `supported_extensions = ['.txt']`
- [ ] Implement `parse()` method:
  - Read file with UTF-8 encoding
  - Handle encoding errors gracefully
  - Clean and return text
- [ ] Implement `extract_metadata()` method:
  - Return minimal metadata (file dates only)
  - No embedded metadata in plain text

### 3.5 HTML Parser Implementation
- [ ] Create `doclint/parsers/html.py`
- [ ] Implement `HTMLParser` class extending `BaseParser`
- [ ] Set `file_type = "html"` and `supported_extensions = ['.html', '.htm']`
- [ ] Implement `parse()` method:
  - Parse HTML with BeautifulSoup (lxml parser)
  - Extract text from body
  - Remove script/style tags
  - Clean and return text
- [ ] Implement `extract_metadata()` method:
  - Extract from meta tags (author, description)
  - Extract title from `<title>` tag
  - Parse date meta tags if present
  - Return DocumentMetadata

### 3.6 DOCX Parser Implementation
- [ ] Create `doclint/parsers/docx.py`
- [ ] Implement `DOCXParser` class extending `BaseParser`
- [ ] Set `file_type = "docx"` and `supported_extensions = ['.docx']`
- [ ] Implement `parse()` method:
  - Open with `python-docx`
  - Extract text from all paragraphs
  - Extract text from tables
  - Join with newlines
  - Clean and return text
- [ ] Implement `extract_metadata()` method:
  - Read core properties (author, title, created, modified)
  - Parse dates from document properties
  - Return DocumentMetadata

### 3.7 Parser Registry
- [ ] Create `doclint/parsers/registry.py`
- [ ] Implement `ParserRegistry` class:
  - Store parsers in a dictionary
  - `register(parser: BaseParser)` method
  - `get_parser(file_path: Path) -> BaseParser` method
  - `can_parse(file_path: Path) -> bool` method
  - `get_supported_extensions() -> List[str]` method
- [ ] Implement auto-registration on import
- [ ] Add default parsers registration

### 3.8 Parser Module Initialization
- [ ] Update `doclint/parsers/__init__.py`:
  - Import all parser classes
  - Export BaseParser and concrete parsers
  - Create default registry instance
  - Auto-register all parsers

### 3.9 Test Fixtures
- [ ] Create `tests/fixtures/` directory structure:
  - `pdfs/`
  - `docx/`
  - `html/`
  - `markdown/`
  - `text/`
- [ ] Create sample PDF file
- [ ] Create sample DOCX file
- [ ] Create sample HTML file
- [ ] Create sample Markdown file with frontmatter
- [ ] Create sample plain text file
- [ ] Create corrupted/malformed file samples

### 3.10 Unit Tests - Base Parser
- [ ] Create `tests/test_parsers/test_base.py`
- [ ] Test `can_parse()` logic
- [ ] Test `clean_text()` function
- [ ] Test abstract methods raise NotImplementedError

### 3.11 Unit Tests - PDF Parser
- [ ] Create `tests/test_parsers/test_pdf.py`
- [ ] Test parsing valid PDF
- [ ] Test metadata extraction
- [ ] Test date parsing
- [ ] Test error handling (encrypted PDF, corrupted file)
- [ ] Test empty PDF

### 3.12 Unit Tests - Markdown Parser
- [ ] Create `tests/test_parsers/test_markdown.py`
- [ ] Test parsing markdown with frontmatter
- [ ] Test parsing markdown without frontmatter
- [ ] Test metadata extraction from YAML
- [ ] Test title extraction from H1

### 3.13 Unit Tests - Text Parser
- [ ] Create `tests/test_parsers/test_text.py`
- [ ] Test parsing UTF-8 text
- [ ] Test encoding error handling
- [ ] Test empty file

### 3.14 Unit Tests - HTML Parser
- [ ] Create `tests/test_parsers/test_html.py`
- [ ] Test text extraction from HTML
- [ ] Test script/style removal
- [ ] Test metadata extraction from meta tags
- [ ] Test malformed HTML handling

### 3.15 Unit Tests - DOCX Parser
- [ ] Create `tests/test_parsers/test_docx.py`
- [ ] Test text extraction
- [ ] Test table text extraction
- [ ] Test metadata extraction
- [ ] Test error handling

### 3.16 Unit Tests - Parser Registry
- [ ] Create `tests/test_parsers/test_registry.py`
- [ ] Test parser registration
- [ ] Test parser lookup by file extension
- [ ] Test can_parse() with various files
- [ ] Test get_supported_extensions()

## Success Criteria
- ✅ All 5 parsers implemented (PDF, DOCX, HTML, MD, TXT)
- ✅ Parser registry working correctly
- ✅ All parser unit tests passing (>95% coverage)
- ✅ Test fixtures created and organized
- ✅ Error handling robust
- ✅ Metadata extraction working for all formats

## Dependencies
- Requires: Core Module (02)
- Required by: Scanner integration, E2E tests

## Notes
- Start with simple parsers (Text, Markdown) to validate architecture
- PDF parsing can be tricky - handle edge cases
- Focus on extracting clean, useful text
- Metadata extraction is best-effort (don't fail if missing)
