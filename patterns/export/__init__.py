"""
Export Patterns - Universal Business Solution Framework

Document generation and data export patterns for professional reports.
Supports multiple formats: CSV, JSON, Excel, Word, HTML, Markdown.

Usage:
```python
from patterns.export import (
    # Report Generation
    ReportGenerator, ReportFormat, ReportSection,
    DataExporter,
    create_analysis_report,

    # Document Building
    DocumentBuilder, DocumentStyle,
    OnePagerBuilder,
    create_proposal_document,
)
```
"""

from .report_generator import (
    ReportGenerator,
    ReportFormat,
    ReportSection,
    ReportMetadata,
    SectionType,
    DataExporter,
    # Factory functions
    create_simple_report,
    create_analysis_report,
    create_executive_summary,
)

from .document_builder import (
    DocumentBuilder,
    DocumentStyle,
    StyleConfig,
    OnePagerBuilder,
    # Factory functions
    create_simple_document,
    create_report_document,
    create_proposal_document,
)

__all__ = [
    # Report Generator
    "ReportGenerator",
    "ReportFormat",
    "ReportSection",
    "ReportMetadata",
    "SectionType",
    "DataExporter",
    "create_simple_report",
    "create_analysis_report",
    "create_executive_summary",

    # Document Builder
    "DocumentBuilder",
    "DocumentStyle",
    "StyleConfig",
    "OnePagerBuilder",
    "create_simple_document",
    "create_report_document",
    "create_proposal_document",
]
