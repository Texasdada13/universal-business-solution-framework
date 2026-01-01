# Universal Business Solution Framework

**Rapidly build enterprise business automation tools using proven patterns and reusable components.**

Built for use with Claude Code to enable lightning-fast development of new business solutions.

---

## Overview

**29 patterns** across **9 categories** - everything you need to build:
- Data monitoring systems
- ETL pipelines
- Alert & notification systems
- Business analysis tools
- Executive dashboards
- Automated scrapers

### Example Projects Built With This Framework:
- **NYPD Procurement Tracker** - Web scraping + change detection + dashboard (4 hours)
- **Application Rationalization Tool** - Data analysis + documentation generation (6 hours)
- **Contract Oversight System** - Scoring + alerts + risk classification
- **Stock Analysis Tool** - ML forecasting + ensemble prediction

---

## Quick Start

```bash
# Create a new project with patterns
python create_project.py --name "My Project" --patterns dashboard,notifications-suite

# Or interactive mode
python create_project.py --interactive

# List all available patterns
python create_project.py --list-patterns
```

---

## Pattern Categories

| Category | Patterns | Description |
|----------|----------|-------------|
| **Core** | 7 | Scraping, change detection, email, scheduler, dashboard, analysis, docs |
| **Scoring** | 4 | Weighted scoring, alert rules, risk classification, benchmarking |
| **ML/AI** | 4 | Feature pipeline, time series, ensemble, model registry |
| **Razors** | 3 | Temporal decay, filter chains, deduplication |
| **Export** | 2 | Report generator, document builder |
| **Notifications** | 3 | Email digest, alert aggregator, notification router |
| **Connectors** | 3 | API client, scraper base, data source registry |
| **Scheduling** | 3 | Job scheduler, pipeline orchestrator, checkpoint manager |

---

## Common Pattern Combinations

### Monitoring System
```bash
python create_project.py --name "Contract Monitor" \
  --patterns scraping,change-detection,notifications-suite,dashboard
```
**Use For**: Track websites for changes, get instant alerts, monitor via dashboard

### ETL Pipeline with Recovery
```bash
python create_project.py --name "Data Pipeline" \
  --patterns connectors-suite,scheduling-suite,export-suite
```
**Use For**: Extract, transform, load data with checkpoint/resume capability

### Business Analysis Tool
```bash
python create_project.py --name "Portfolio Analyzer" \
  --patterns scoring-suite,razors-suite,dashboard
```
**Use For**: Score, classify, and analyze business data

### ML Forecasting System
```bash
python create_project.py --name "Demand Forecaster" \
  --patterns ml-suite,dashboard,export-suite
```
**Use For**: Time series forecasting with ensemble models

### Alert & Notification System
```bash
python create_project.py --name "Alert Center" \
  --patterns notifications-suite,scheduling-suite
```
**Use For**: Aggregate alerts, dedupe, route to Slack/Teams/Email

---

## Documentation

| Document | Description |
|----------|-------------|
| [Quick Start Guide](docs/QUICK-START.md) | Get up and running in 5 minutes |
| [Pattern Reference](docs/PATTERNS.md) | Detailed documentation for all 29 patterns |
| [Examples](docs/EXAMPLES.md) | Real-world usage examples |
| [API Reference](docs/API.md) | Complete API documentation |

---

## Project Structure

```
universal-business-solution-framework/
├── patterns/
│   ├── core/           # Base patterns (scraping, email, etc.)
│   ├── scoring/        # Scoring & risk patterns
│   ├── ml/             # Machine learning patterns
│   ├── razors/         # Analysis & filtering patterns
│   ├── export/         # Report & document generation
│   ├── notifications/  # Alert & notification patterns
│   ├── connectors/     # Data connection patterns
│   └── scheduling/     # Job & pipeline orchestration
├── templates/          # Project templates
├── examples/           # Example projects
├── docs/               # Documentation
├── create_project.py   # Project generator
└── requirements.txt    # Dependencies
```

---

## Key Features

### Zero-Dependency Patterns
Most patterns work with Python standard library only. Optional dependencies (pandas, beautifulsoup4) enhance functionality.

### Factory Functions
Every pattern includes factory functions for common use cases:
```python
from patterns.scheduling import create_simple_scheduler
from patterns.notifications import create_slack_router
from patterns.connectors import create_gentle_scraper
```

### Fluent APIs
Builder patterns for complex configurations:
```python
from patterns.connectors import APIClientBuilder

client = (APIClientBuilder("https://api.example.com")
    .with_bearer_token("token")
    .with_rate_limit(10)
    .with_retry(max_retries=3)
    .build())
```

### Composable Design
Patterns work together seamlessly:
```python
from patterns.connectors import create_gentle_scraper
from patterns.notifications import create_standard_aggregator, create_slack_router

# Scrape -> Analyze -> Alert -> Notify
scraper = create_gentle_scraper()
aggregator = create_standard_aggregator()
router = create_slack_router(webhook_url)

data = scraper.get_json("https://api.example.com/data")
aggregator.add_alert("New Data", f"Found {len(data)} items")

for alert in aggregator.get_pending_alerts():
    router.send(Notification(title=alert.title, message=alert.message))
```

---

## Version History

| Version | Focus | Total Patterns |
|---------|-------|----------------|
| **v1.4** | Infrastructure (Notifications, Connectors, Scheduling) | 29 |
| v1.3 | Analysis (Razors, Export) | 20 |
| v1.2 | ML/AI (Features, Forecasting, Ensemble, Registry) | 15 |
| v1.1 | Scoring (Weighted, Alerts, Risk, Benchmarks) | 11 |
| v1.0 | Core patterns | 7 |

---

## Requirements

- **Python 3.8+**
- **Dependencies**: Automatically generated per project based on selected patterns
- **Optional**: Email account for alerts, web hosting for dashboards

---

## Best Practices

### 1. Start With Business Value
- What problem does this solve?
- What's the ROI?
- Who are the users?

### 2. Use Suite Patterns
Suite patterns include all related patterns:
- `scoring-suite` - All scoring patterns
- `ml-suite` - All ML patterns
- `notifications-suite` - All notification patterns
- `connectors-suite` - All connector patterns
- `scheduling-suite` - All scheduling patterns

### 3. Leverage Factory Functions
```python
# Instead of configuring from scratch
from patterns.notifications import create_standard_aggregator
aggregator = create_standard_aggregator(max_per_hour=20)
```

### 4. Security First
- Environment variables for secrets
- Never commit credentials
- Validate all inputs

---

## How to Use With Claude Code

### Method 1: Direct Prompt
```
Open Claude Code in this framework directory and say:

"Using this framework, build a tool that:
- Scrapes [URL] for [data]
- Detects when new [items] appear
- Sends Slack notifications
- Shows a dashboard

Use patterns: scraping + change-detection + notifications-suite + dashboard"
```

### Method 2: Generate Then Customize
```bash
# Generate project structure
python create_project.py --name "NYC Building Permits" --patterns scraping,notifications-suite

# Open in Claude Code
cd nyc-building-permits

# Prompt Claude:
"Customize this project to scrape NYC building permits from [URL].
The permit fields are: address, type, date, status.
Send Slack alerts when new permits are added."
```

---

## Troubleshooting

### Import errors
```bash
pip install -r requirements.txt
```

### Pattern not found
```bash
python create_project.py --list-patterns
```

### Project generation fails
- Check Python version (3.8+)
- Ensure write permissions
- Verify pattern names are correct

---

## License

MIT License - Free to use and modify

---

**Built to accelerate business solution development from weeks to hours.**
