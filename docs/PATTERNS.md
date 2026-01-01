# Pattern Reference

Complete documentation for all 29 patterns in the Universal Business Solution Framework.

---

## Table of Contents

1. [Core Patterns](#core-patterns)
2. [Scoring Patterns](#scoring-patterns)
3. [ML/AI Patterns](#mlai-patterns)
4. [Razors Patterns](#razors-patterns)
5. [Export Patterns](#export-patterns)
6. [Notifications Patterns](#notifications-patterns)
7. [Connectors Patterns](#connectors-patterns)
8. [Scheduling Patterns](#scheduling-patterns)

---

## Core Patterns

### 1. Web Scraping
**File**: `patterns/scraping_patterns.py`

Extract data from websites using BeautifulSoup.

```python
from patterns.core import WebScraper

scraper = WebScraper()
data = scraper.scrape_table("https://example.com/data")
scraper.export_csv(data, "output.csv")
```

**Features**:
- HTML parsing with BeautifulSoup
- Table extraction
- CSV/JSON export
- Error handling and retry

---

### 2. Change Detection
**File**: `patterns/change_detection_patterns.py`

Track changes in data over time using MD5 fingerprinting.

```python
from patterns.core import ChangeDetector

detector = ChangeDetector("baseline.json")
changes = detector.detect_changes(current_data)

print(f"New: {len(changes['new'])}")
print(f"Removed: {len(changes['removed'])}")
print(f"Modified: {len(changes['modified'])}")
```

**Features**:
- MD5 hash-based fingerprinting
- Baseline comparison
- History tracking
- New/removed/modified detection

---

### 3. Email Alerts
**File**: `patterns/email_patterns.py`

Send professional HTML email notifications.

```python
from patterns.core import EmailSender

sender = EmailSender(smtp_host, smtp_user, smtp_pass)
sender.send_html_email(
    to="user@example.com",
    subject="Alert: New Items Found",
    html_content=html_template
)
```

**Features**:
- HTML + plain-text emails
- SMTP integration
- Template system
- Attachment support

---

### 4. Task Scheduling
**File**: `patterns/scheduler_patterns.py`

Run tasks automatically at intervals.

```python
from patterns.core import TaskScheduler

scheduler = TaskScheduler()
scheduler.add_task("check_data", check_function, interval_minutes=5)
scheduler.start()
```

**Features**:
- Configurable intervals
- Background operation
- Error handling
- Multi-task support

---

### 5. Web Dashboard
**Pattern**: `dashboard`

Flask-based real-time monitoring interface.

```python
# web/app.py
from flask import Flask, render_template, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/data')
def get_data():
    return jsonify(load_data())
```

**Features**:
- Flask backend
- REST APIs
- Chart.js visualizations
- Auto-refresh
- Responsive design

---

### 6. Data Analysis
**Pattern**: `analysis`

Process and analyze data with pandas.

**Features**:
- DataFrame operations
- Excel/CSV processing
- Statistical analysis
- Data validation

---

### 7. Documentation Generation
**Pattern**: `documentation`

Create professional documents.

**Features**:
- Markdown generation
- Word document creation
- Problem/solution statements
- Executive summaries

---

## Scoring Patterns

### 8. Weighted Scoring Engine
**File**: `patterns/scoring/weighted_scoring.py`

Multi-component weighted scoring with grades and risk levels.

```python
from patterns.scoring import WeightedScorer, ScoringComponent

scorer = WeightedScorer([
    ScoringComponent("quality", weight=0.4),
    ScoringComponent("timeliness", weight=0.3),
    ScoringComponent("cost", weight=0.3)
])

result = scorer.score({
    "quality": 85,
    "timeliness": 90,
    "cost": 75
})

print(f"Score: {result.total_score}")  # 83.5
print(f"Grade: {result.grade}")        # B
```

**Features**:
- Configurable weights
- Grade mapping (A-F)
- Risk level classification
- Component breakdown

---

### 9. Alert Rules Engine
**File**: `patterns/scoring/alert_rules_engine.py`

Declarative rule-based alert generation.

```python
from patterns.scoring import AlertRulesEngine, AlertRule, AlertSeverity

engine = AlertRulesEngine([
    AlertRule(
        name="low_score",
        condition=lambda x: x["score"] < 50,
        severity=AlertSeverity.HIGH,
        message="Score below threshold"
    )
])

alerts = engine.evaluate({"score": 45, "status": "active"})
```

**Features**:
- Declarative rules
- Multiple severity levels
- Conditional logic
- Alert aggregation

---

### 10. Risk Classification
**File**: `patterns/scoring/risk_classification.py`

Convert scores to risk levels with configurable thresholds.

```python
from patterns.scoring import RiskClassifier, RiskLevel

classifier = RiskClassifier(
    thresholds={"critical": 20, "high": 40, "medium": 60, "low": 80}
)

risk = classifier.classify(35)  # RiskLevel.HIGH
```

**Features**:
- Configurable thresholds
- Risk level enums
- Trend analysis
- Batch classification

---

### 11. KPI Benchmarking
**File**: `patterns/scoring/benchmark_engine.py`

Compare KPIs against benchmarks with gap analysis.

```python
from patterns.scoring import BenchmarkEngine

engine = BenchmarkEngine({
    "response_time_ms": {"target": 200, "warning": 500},
    "uptime_percent": {"target": 99.9, "warning": 99.0}
})

result = engine.evaluate({"response_time_ms": 350, "uptime_percent": 99.5})
print(result.gaps)  # Shows gaps from targets
```

**Features**:
- Target/warning thresholds
- Gap analysis
- Performance grades
- Trend tracking

---

## ML/AI Patterns

### 12. Feature Engineering Pipeline
**File**: `patterns/ml/feature_pipeline.py`

Generate technical indicators and features for ML models.

```python
from patterns.ml import FeaturePipeline, FeatureCategory

pipeline = FeaturePipeline(categories=[
    FeatureCategory.PRICE,
    FeatureCategory.MOMENTUM,
    FeatureCategory.VOLATILITY
])

features_df = pipeline.generate_features(df)
```

**Features**:
- 50+ technical indicators (RSI, MACD, Bollinger Bands)
- Momentum, volatility, trend features
- Statistical features (skewness, kurtosis)
- Custom feature registration

---

### 13. Time Series Forecasting
**File**: `patterns/ml/time_series_models.py`

Forecast future values from historical data.

```python
from patterns.ml import TimeSeriesForecaster, ForecastMethod

forecaster = TimeSeriesForecaster(method=ForecastMethod.ENSEMBLE)
forecaster.fit(df['close'])

result = forecaster.predict_with_intervals(horizon=30)
print(f"Predictions: {result.predictions}")
print(f"Confidence: {result.confidence_level}")
```

**Features**:
- Multiple methods (Moving Average, Exponential Smoothing, ARIMA)
- Ensemble forecasting
- Confidence intervals
- Automatic seasonality detection

---

### 14. Ensemble Predictor
**File**: `patterns/ml/ensemble_predictor.py`

Combine multiple models for better predictions.

```python
from patterns.ml import EnsemblePredictor, EnsembleMethod

ensemble = EnsemblePredictor(method=EnsembleMethod.WEIGHTED_AVERAGE)
ensemble.register_model('model_a', model_a, weight=0.5)
ensemble.register_model('model_b', model_b, weight=0.5)

result = ensemble.predict_with_uncertainty(X_test)
```

**Features**:
- Multiple ensemble methods (Weighted, Stacking, Voting)
- Uncertainty quantification
- Auto-weighting
- Model enable/disable

---

### 15. Model Registry
**File**: `patterns/ml/model_registry.py`

Model versioning, storage, and lifecycle management.

```python
from patterns.ml import ModelRegistry, ModelStage

registry = ModelRegistry(storage_path='models/')

version = registry.register(
    name='price_predictor',
    model=trained_model,
    metrics={'rmse': 0.05}
)

registry.transition_stage('price_predictor', version.version, ModelStage.PRODUCTION)
model = registry.load('price_predictor', stage=ModelStage.PRODUCTION)
```

**Features**:
- Automatic versioning
- Lifecycle stages (Development, Staging, Production, Archived)
- Model comparison
- Tag-based search

---

## Razors Patterns

### 16. Temporal Decay Scoring
**File**: `patterns/razors/temporal_decay.py`

Time-based scoring where freshness matters.

```python
from patterns.razors import TemporalDecay, create_recency_scorer

decay = create_recency_scorer()
score = decay.score(days_old=25)  # Returns 75

# Expiration tracking
from patterns.razors import ExpirationTracker
tracker = ExpirationTracker(warning_days=90, critical_days=30)
status = tracker.check(contract_end_date)
```

**Features**:
- Bracket-based scoring (7 days = 100, 14 days = 90, etc.)
- Multiple decay functions (linear, exponential, sigmoid)
- Expiration tracking with urgency levels
- Freshness labels

---

### 17. Filter Chains
**File**: `patterns/razors/filter_chains.py`

Composable filter logic with fluent API.

```python
from patterns.razors import FilterChain, create_priority_filter

# Fluent API
filtered = (FilterChain()
    .where('status').equals('Active')
    .where('score').greater_than(50)
    .exclude('category').is_in(['spam', 'test'])
    .apply(items))

# Priority classification
priority_filter = create_priority_filter(urgent=85, high=70, medium=40)
classified = priority_filter.classify(items)
```

**Features**:
- Fluent API for building filters
- AND/OR logic
- Negative filters (exclusion lists)
- Threshold-based classification

---

### 18. Deduplication & Change Detection
**File**: `patterns/razors/deduplication.py`

Multi-key deduplication and change tracking.

```python
from patterns.razors import Deduplicator, ChangeDetector

# Deduplication
dedup = Deduplicator(key_fields=['agency', 'title'])
unique_items = dedup.deduplicate(all_items)

# Change detection
detector = ChangeDetector(
    key_fields=['contract_id'],
    compare_fields=['status', 'amount']
)
changes = detector.detect_changes(new_data, existing_data)
```

**Features**:
- Multi-key deduplication (primary + fallback keys)
- Change detection (new, modified, removed)
- Fingerprint/hash generation
- Configurable matching strategies

---

## Export Patterns

### 19. Report Generator
**File**: `patterns/export/report_generator.py`

Multi-format data exports and reports.

```python
from patterns.export import ReportGenerator, ReportFormat

report = ReportGenerator(title="Q4 Performance Report")
report.add_metrics("Key Metrics", {"Total Revenue": "$1.2M", "Growth": "15%"})
report.add_table("Sales by Region", sales_data)
report.add_list("Key Achievements", achievements)

report.export("report.xlsx", ReportFormat.EXCEL)
report.export("report.html", ReportFormat.HTML)
report.export_all("report")  # All formats
```

**Features**:
- Multiple formats (CSV, JSON, Excel, HTML, Markdown)
- Structured sections
- Professional styling
- Batch export

---

### 20. Document Builder
**File**: `patterns/export/document_builder.py`

Professional Word/HTML documents with styling.

```python
from patterns.export import DocumentBuilder, DocumentStyle

doc = DocumentBuilder(title="Project Proposal", style=DocumentStyle.CORPORATE)
doc.add_heading("Executive Summary", level=1)
doc.add_paragraph("This proposal outlines...")
doc.add_bullet_list(["Benefit 1", "Benefit 2", "Benefit 3"])
doc.add_table(data, headers=['Phase', 'Duration', 'Deliverables'])
doc.save("proposal.docx")
```

**Features**:
- Word document generation
- Multiple style presets (Corporate, Executive, Technical)
- Tables, lists, images
- Fallback to Markdown/HTML

---

## Notifications Patterns

### 21. Email Digest Builder
**File**: `patterns/notifications/email_digest.py`

Professional HTML email templates with themes.

```python
from patterns.notifications import EmailDigestBuilder, DigestTheme

digest = EmailDigestBuilder("Daily Summary", theme=DigestTheme.PROFESSIONAL)
digest.add_summary({"New Alerts": 12, "Resolved": 5})
digest.add_alert_section("Critical Alerts", critical_alerts)
digest.add_table("Recent Items", items_data)
digest.add_action_button("View Dashboard", "https://app.example.com")

html, text = digest.build()
```

**Features**:
- Multiple themes (Professional, Alert, Success, Minimal, Dark)
- Summary cards
- Alert sections with severity styling
- Action buttons
- Plain-text fallback

---

### 22. Alert Aggregator
**File**: `patterns/notifications/alert_aggregator.py`

Deduplicate, rate-limit, and group alerts.

```python
from patterns.notifications import AlertAggregator, AlertPriority

aggregator = AlertAggregator(
    dedup_window_minutes=60,
    max_alerts_per_hour=20,
    quiet_hours=(22, 7)
)

aggregator.add_alert("Server Down", "Web server not responding", AlertPriority.CRITICAL)

for alert in aggregator.get_pending_alerts():
    send_notification(alert)
    aggregator.mark_sent(alert.id)
```

**Features**:
- Deduplication within time window
- Rate limiting per priority
- Quiet hours suppression
- Priority escalation
- Alert grouping

---

### 23. Notification Router
**File**: `patterns/notifications/notification_router.py`

Route notifications to multiple channels.

```python
from patterns.notifications import NotificationRouter, SlackHandler, Channel

router = NotificationRouter()
router.add_channel(Channel.SLACK, SlackHandler(webhook_url))
router.add_channel(Channel.TEAMS, TeamsHandler(teams_webhook))
router.set_default_channels([Channel.SLACK])

# Add routing rules
from patterns.notifications import RoutingRule, NotificationPriority
router.add_rule(RoutingRule(
    condition=lambda n: n.priority == NotificationPriority.CRITICAL,
    channels=[Channel.SLACK, Channel.TEAMS],
    immediate=True
))

notification = Notification(title="Alert", message="Issue detected")
results = router.send(notification)
```

**Features**:
- Multiple channels (Email, Slack, Teams, Webhook)
- Routing rules with conditions
- Fallback channels
- Retry logic
- Delivery tracking

---

## Connectors Patterns

### 24. API Client
**File**: `patterns/connectors/api_client.py`

Robust API client with rate limiting and retry.

```python
from patterns.connectors import APIClientBuilder

client = (APIClientBuilder("https://api.example.com")
    .with_bearer_token("token")
    .with_rate_limit(10)
    .with_retry(max_retries=3)
    .with_cache(ttl=300)
    .build())

users = client.get("/users")
result = client.post("/users", json={"name": "John"})
all_items = client.get_all_pages("/items", page_param="page")
```

**Features**:
- Multiple auth methods (Bearer, API Key, Basic)
- Token bucket rate limiting
- Automatic retry with backoff
- Response caching
- Pagination support

---

### 25. Scraper Base
**File**: `patterns/connectors/scraper_base.py`

Web scraping with anti-detection features.

```python
from patterns.connectors import ScraperBase, ScraperConfig

scraper = ScraperBase(ScraperConfig(
    rate_limit_seconds=2.0,
    rotate_user_agents=True,
    random_delay=True
))

result = scraper.get("https://example.com")
soup = scraper.get_soup("https://example.com")

# Session-based scraping
with scraper.session() as session:
    session.get("https://example.com/login")
    session.post("https://example.com/login", data={"user": "x"})
    data = session.get("https://example.com/protected")
```

**Features**:
- Session management with cookies
- User agent rotation
- Random delay variance
- Proxy support
- Retry with backoff

---

### 26. Data Source Registry
**File**: `patterns/connectors/data_source_registry.py`

Unified interface for multiple data sources.

```python
from patterns.connectors import DataSourceRegistry, APISource, FileSource

registry = DataSourceRegistry()
registry.register("users_api", APISource(base_url="https://api.example.com"))
registry.register("backup", FileSource("data/backup.json"))

result = registry.fetch("users_api", "/users")
health = registry.health_check_all()

# With caching
from patterns.connectors import CachedSource
cached_api = CachedSource(APISource(base_url="..."), ttl_seconds=300)
registry.register("cached_users", cached_api)
```

**Features**:
- Multiple source types (API, File, Memory)
- Health monitoring
- Access tracking
- Caching wrapper
- Fallback sources

---

## Scheduling Patterns

### 27. Job Scheduler
**File**: `patterns/scheduling/job_scheduler.py`

Cron-like job scheduling with dependencies.

```python
from patterns.scheduling import JobScheduler, Job, Schedule

scheduler = JobScheduler(max_concurrent=5)

scheduler.add_job(Job(
    name="daily_report",
    func=generate_report,
    schedule=Schedule.daily(hour=8, minute=0)
))

scheduler.add_job(Job(
    name="hourly_sync",
    func=sync_data,
    schedule=Schedule.hourly()
))

scheduler.add_job(Job(
    name="every_5_min",
    func=check_alerts,
    schedule=Schedule.interval(minutes=5)
))

scheduler.start()
```

**Features**:
- Interval, daily, hourly, cron scheduling
- Job dependencies
- Retry with backoff
- Execution history
- Concurrent job limits

---

### 28. Pipeline Orchestrator
**File**: `patterns/scheduling/pipeline_orchestrator.py`

DAG-based task pipelines with parallel execution.

```python
from patterns.scheduling import Pipeline, Task

pipeline = Pipeline("data_pipeline", max_parallel=4)

pipeline.add_task(Task("extract", extract_func))
pipeline.add_task(Task("transform", transform_func, depends_on=["extract"]))
pipeline.add_task(Task("load", load_func, depends_on=["transform"]))

result = pipeline.run()
print(f"Status: {result.status}")
print(f"Duration: {result.duration_ms}ms")
print(pipeline.visualize())  # ASCII DAG
```

**Features**:
- Dependency-based execution order
- Parallel execution of independent tasks
- Retry logic per task
- Shared context
- Visual DAG representation

---

### 29. Checkpoint Manager
**File**: `patterns/scheduling/checkpoint_manager.py`

Resume long-running processes from failure.

```python
from patterns.scheduling import CheckpointManager

checkpoint = CheckpointManager("data_processing", storage_path="checkpoints/")

# Resume or start fresh
state = checkpoint.load() or {"processed": 0}

for i, item in enumerate(items):
    if i < state["processed"]:
        continue  # Skip already processed

    process(item)
    state["processed"] = i + 1

    if i % 100 == 0:
        checkpoint.save(state)

checkpoint.complete(state)
```

**Features**:
- Save and load state
- Automatic versioning
- Progress tracking
- Checkpoint history
- Incremental processing

---

## Pattern Suites

For convenience, suite patterns bundle related patterns together:

| Suite | Includes |
|-------|----------|
| `scoring-suite` | weighted-scoring, alert-rules, risk-classification, benchmarking |
| `ml-suite` | feature-pipeline, time-series, ensemble, model-registry |
| `razors-suite` | temporal-decay, filter-chains, deduplication |
| `export-suite` | report-generator, document-builder |
| `notifications-suite` | email-digest, alert-aggregator, notification-router |
| `connectors-suite` | api-client, scraper-base, data-source-registry |
| `scheduling-suite` | job-scheduler, pipeline-orchestrator, checkpoint-manager |

```bash
# Use suites for quick project setup
python create_project.py --name "Full Pipeline" \
  --patterns connectors-suite,scheduling-suite,notifications-suite
```
