# Universal Business Solution Framework

**Rapidly build enterprise business automation tools using proven patterns and reusable components.**

Built for use with Claude Code to enable lightning-fast development of new business solutions.

---

## What Is This?

A framework of battle-tested patterns, templates, and components extracted from real enterprise projects. Instead of building from scratch, assemble pre-built components to create new tools in hours instead of weeks.

### Example Projects Built With This Framework:
- **NYPD Procurement Tracker** - Web scraping + change detection + dashboard (4 hours)
- **Application Rationalization Tool** - Data analysis + documentation generation (6 hours)
- **Capital Projects Planner** - Data processing + reporting (5 hours)

---

## Quick Start (3 Steps)

### Step 1: Clone or Download Framework
```bash
cd C:\Users\dada_\OneDrive\Documents
# Framework is already here: universal-business-solution-framework/
```

### Step 2: Create a New Project
```bash
cd universal-business-solution-framework

# Interactive mode (recommended for first time)
python create_project.py --interactive

# Or specify directly
python create_project.py --name "My Business Tool" --patterns scraping,change-detection,email,dashboard
```

### Step 3: Customize and Run
```bash
cd my-business-tool
pip install -r requirements.txt
# Edit src/ files for your specific use case
python web/app.py  # If using dashboard
```

---

## Available Patterns

### 1. Web Scraping
Extract data from websites using BeautifulSoup
- HTML parsing
- Table extraction
- CSV/JSON export
- Error handling

### 2. Change Detection
Track what's new or changed over time
- MD5 hash-based fingerprinting
- Baseline comparison
- History tracking
- New/removed item detection

### 3. Email Alerts
Send professional email notifications
- HTML + plain-text emails
- SMTP integration (Gmail, Outlook, custom)
- Template system
- Professional formatting

### 4. Web Dashboard
Real-time monitoring interface
- Flask backend
- REST APIs
- Chart.js visualizations
- Auto-refresh
- Responsive design

### 5. Task Scheduling
Automated task execution
- Configurable intervals
- Background operation
- Error handling
- Multi-task support

### 6. Data Analysis
Process and analyze data
- Pandas DataFrame operations
- Excel/CSV processing
- Statistical analysis
- Data validation

### 7. Documentation Generation
Create professional docs
- Markdown generation
- Word document creation
- Problem/solution statements
- Executive summaries

---

## Common Pattern Combinations

### Monitoring System
**Patterns**: `scraping,change-detection,email,dashboard`
**Use For**: Track websites for changes, get instant alerts, monitor via dashboard
**Time**: 2-4 hours
**Example**: NYPD Procurement Tracker

### Business Analysis Tool
**Patterns**: `analysis,documentation,dashboard`
**Use For**: Analyze datasets, generate reports, executive dashboards
**Time**: 4-6 hours
**Example**: Application Rationalization Tool

### Data Pipeline
**Patterns**: `scraping,scheduler,email`
**Use For**: Extract data regularly, process, notify on completion
**Time**: 3-5 hours

### Alert System
**Patterns**: `change-detection,email,scheduler`
**Use For**: Monitor for specific events, instant notifications
**Time**: 2-3 hours

---

## How to Use With Claude Code

### Method 1: Direct Prompt
```
Open Claude Code in this framework directory and say:

"Using this framework, build a tool that:
- Scrapes [URL] for [data]
- Detects when new [items] appear
- Sends email alerts
- Shows a dashboard

Use patterns: scraping + change-detection + email + dashboard"
```

### Method 2: Generate Then Customize
```bash
# Generate project structure
python create_project.py --name "NYC Building Permits" --patterns scraping,change-detection,email

# Open in Claude Code
cd nyc-building-permits

# Prompt Claude:
"Customize this project to scrape NYC building permits from [URL].
The permit fields are: address, type, date, status.
Send alerts when new permits are added."
```

---

## Project Structure

```
universal-business-solution-framework/
├── .claude/
│   └── framework-context.md       # Explains framework to Claude Code
├── patterns/
│   ├── scraping_patterns.py       # Web scraping components
│   ├── change_detection_patterns.py  # Change tracking
│   ├── email_patterns.py          # Email alerts
│   ├── scheduler_patterns.py      # Task scheduling
│   └── ... (more patterns)
├── templates/
│   ├── data-pipeline/             # Pipeline templates
│   ├── web-dashboard/             # Dashboard templates
│   └── ... (more templates)
├── examples/
│   └── Links to completed projects
├── docs/
│   └── Framework documentation
├── create_project.py              # Project generator
└── README.md                      # This file
```

---

## Commands Reference

### List Available Patterns
```bash
python create_project.py --list-patterns
```

### Create Project (Interactive)
```bash
python create_project.py --interactive
```

### Create Project (Direct)
```bash
python create_project.py --name "Project Name" --patterns scraping,email,dashboard
```

### Common Pattern Combinations
```bash
# Monitoring system
--patterns scraping,change-detection,email,dashboard,scheduler

# Business analysis
--patterns analysis,documentation,dashboard

# Data pipeline
--patterns scraping,scheduler,email

# Alert system
--patterns change-detection,email,scheduler
```

---

## Example Projects (Reference)

### 1. NYPD Procurement Tracker
**Location**: `C:\Users\dada_\OneDrive\Documents\nypd-procurement-tracker\`
**Patterns**: Scraping + Change Detection + Email + Dashboard + Scheduler
**Time to Build**: ~4 hours
**Business Value**: First-mover advantage on opportunities

**Key Features**:
- Scrapes 76 procurement opportunities
- Detects new opportunities instantly
- Sends HTML email alerts
- Real-time web dashboard with charts
- Auto-refresh every 5 minutes

### 2. Application Rationalization Tool
**Location**: `C:\Users\dada_\OneDrive\Documents\application-rationalization-tool\`
**Patterns**: Data Analysis + Documentation + Dashboard
**Time to Build**: ~6 hours
**Business Value**: 23x ROI in Year 1

**Key Features**:
- Analyzes application portfolios
- Identifies redundancies
- Generates comprehensive documentation
- Word document generation

### 3. Capital Projects Planner
**Location**: `C:\Users\dada_\OneDrive\Documents\application-rationalization-tool\capital_projects\`
**Patterns**: Data Processing + Analysis + Documentation
**Time to Build**: ~5 hours
**Business Value**: 44x ROI in Year 1

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

### 2. Use Modular Design
- Each pattern is independent
- Easy to add/remove features
- Test components separately

### 3. Document Everything
- Use generated README as starting point
- Add setup guides
- Include inline comments

### 4. Security First
- Environment variables for secrets
- Never commit credentials
- Validate all inputs

---

## Extending the Framework

### Add New Patterns

1. Create pattern file in `patterns/`
2. Document in `.claude/framework-context.md`
3. Add to `create_project.py` AVAILABLE_PATTERNS
4. Test with example project

### Add New Templates

1. Create template structure in `templates/`
2. Add logic to `create_project.py`
3. Document usage

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

## Support

**Framework Issues**: Check `.claude/framework-context.md` for detailed pattern documentation

**Pattern Examples**: Review example projects in `examples/` links

**Claude Code**: Reference `.claude/framework-context.md` when prompting Claude

---

## Version

**v1.0** - Initial framework release
- 7 core patterns
- 3 reference projects
- Project generator
- Complete documentation

---

## License

MIT License - Free to use and modify

---

**Built to accelerate business solution development from weeks to hours.**
