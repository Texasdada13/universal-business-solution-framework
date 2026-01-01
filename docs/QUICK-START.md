# Quick Start Guide

Get up and running with the Universal Business Solution Framework in 5 minutes.

---

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

---

## Step 1: Navigate to the Framework

```bash
cd C:\Users\dada_\OneDrive\Documents\universal-business-solution-framework
```

---

## Step 2: Create Your First Project

### Option A: Interactive Mode (Recommended for Beginners)

```bash
python create_project.py --interactive
```

Follow the prompts:
1. Enter project name
2. Select patterns from the list
3. Confirm creation

### Option B: Command Line

```bash
python create_project.py --name "My Project" --patterns dashboard,notifications-suite
```

### Option C: List Available Patterns First

```bash
python create_project.py --list-patterns
```

---

## Step 3: Install Dependencies

```bash
cd my-project
pip install -r requirements.txt
```

---

## Step 4: Run Your Project

### If you have a dashboard:
```bash
python web/app.py
# Open http://localhost:5001 in your browser
```

### If you have a scheduler:
```bash
python src/main.py
```

---

## Common Project Templates

### 1. Website Monitor

Monitor a website for changes and get notifications:

```bash
python create_project.py --name "Site Monitor" \
  --patterns scraping,change-detection,notifications-suite,dashboard
```

**What you get:**
- Web scraper with rate limiting
- Change detection with fingerprinting
- Email/Slack notifications
- Real-time dashboard

### 2. Data Pipeline

ETL pipeline with checkpoint/resume:

```bash
python create_project.py --name "Data Pipeline" \
  --patterns connectors-suite,scheduling-suite,export-suite
```

**What you get:**
- API client with retry logic
- Job scheduler with dependencies
- Checkpoint manager for resume
- Multi-format report export

### 3. Alert System

Centralized alert management:

```bash
python create_project.py --name "Alert Center" \
  --patterns notifications-suite,scheduling-suite,dashboard
```

**What you get:**
- Alert aggregation and deduplication
- Rate limiting and quiet hours
- Multi-channel routing (Slack, Teams, Email)
- Scheduled checks

### 4. Business Analyzer

Score and analyze business data:

```bash
python create_project.py --name "Portfolio Analyzer" \
  --patterns scoring-suite,razors-suite,export-suite,dashboard
```

**What you get:**
- Weighted scoring engine
- Risk classification
- Filter chains for data analysis
- Professional reports

---

## Using Patterns Directly

You can also use patterns directly without the project generator:

```python
# Import from patterns
from patterns.notifications import EmailDigestBuilder, DigestTheme
from patterns.connectors import create_gentle_scraper
from patterns.scheduling import create_simple_scheduler

# Build email digest
digest = EmailDigestBuilder("Daily Report", theme=DigestTheme.PROFESSIONAL)
digest.add_summary({"Items": 100, "Alerts": 5})
html, text = digest.build()

# Create scraper with rate limiting
scraper = create_gentle_scraper(delay=2.0)
data = scraper.get("https://example.com/api/data")

# Schedule jobs
scheduler = create_simple_scheduler()
scheduler.add_job(Job(name="check", func=check_func, schedule=Schedule.hourly()))
```

---

## Project Structure After Generation

```
my-project/
├── src/
│   ├── notifications/    # If notifications-suite selected
│   ├── connectors/       # If connectors-suite selected
│   ├── scheduling/       # If scheduling-suite selected
│   └── main.py
├── web/                  # If dashboard selected
│   ├── app.py
│   ├── templates/
│   └── static/
├── data/
├── output/
├── docs/
├── tests/
├── requirements.txt
├── .env.example
└── README.md
```

---

## Next Steps

1. **Customize the code** - Edit `src/` files for your specific use case
2. **Configure settings** - Copy `.env.example` to `.env` and set your values
3. **Add tests** - Write tests in `tests/` directory
4. **Deploy** - See deployment guides for your target platform

---

## Common Tasks

### Add a new data source

```python
from patterns.connectors import DataSourceRegistry, APISource

registry = DataSourceRegistry()
registry.register("my_api", APISource(
    base_url="https://api.example.com",
    auth_token="your-token"
))

data = registry.fetch("my_api", "/endpoint")
```

### Schedule a recurring job

```python
from patterns.scheduling import JobScheduler, Job, Schedule

scheduler = JobScheduler()
scheduler.add_job(Job(
    name="hourly_check",
    func=your_function,
    schedule=Schedule.hourly(minute=0)
))
scheduler.start()
```

### Send a Slack notification

```python
from patterns.notifications import create_slack_router, Notification

router = create_slack_router("https://hooks.slack.com/services/...")
router.send(Notification(
    title="Alert",
    message="Something happened!",
    priority=NotificationPriority.HIGH
))
```

### Generate a report

```python
from patterns.export import ReportGenerator, ReportFormat

report = ReportGenerator("Monthly Report")
report.add_metrics("Summary", {"Total": 100, "Active": 85})
report.add_table("Details", data_list)
report.export_all("output/report")  # Creates .xlsx, .html, .md, .json
```

---

## Troubleshooting

### ImportError: No module named 'patterns'

Make sure you're running from the framework directory or the pattern files are in your project's `src/` folder.

### Pattern not found

```bash
python create_project.py --list-patterns
```

### Permission denied

Run with appropriate permissions or check file/folder access.

### Port already in use (dashboard)

Edit `web/app.py` and change the port number:
```python
app.run(port=5002)  # Change from 5001
```

---

## Getting Help

1. Check [Pattern Reference](PATTERNS.md) for detailed documentation
2. Review [Examples](EXAMPLES.md) for real-world usage
3. Look at `.claude/framework-context.md` for the complete framework guide
