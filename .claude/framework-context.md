# Universal Business Solution Framework

## Overview

This framework enables rapid development of enterprise business automation tools by providing battle-tested patterns, reusable components, and templates. It's specifically designed to work with Claude Code for fast implementation of new business ideas.

**Location**: `C:\Users\dada_\OneDrive\Documents\universal-business-solution-framework\`

## Framework Philosophy

Every business automation tool typically combines these core capabilities:
1. **Data Collection** - Getting data from somewhere (scraping, APIs, files, databases)
2. **Data Processing** - Transforming, analyzing, validating data
3. **Change Detection** - Identifying what's new or different
4. **Storage** - Persisting data (JSON, CSV, databases)
5. **Notifications** - Alerting stakeholders (email, Slack, SMS)
6. **Visualization** - Dashboards and reports
7. **Automation** - Scheduling and background processing

This framework provides proven implementations of each capability that can be mixed and matched.

---

## Core Patterns Library

### Pattern 1: Web Scraping
**File**: `patterns/scraping_patterns.py`
**Use When**: Need to extract data from websites
**Features**:
- BeautifulSoup4 HTML parsing
- Requests with retry logic
- Table extraction (converts HTML tables to Python dicts)
- Error handling and logging
- CSV/JSON export

**Example Projects Using This**:
- NYPD Procurement Tracker (scrapes procurement opportunities)

### Pattern 2: Change Detection
**File**: `patterns/change_detection_patterns.py`
**Use When**: Need to track what's new or changed over time
**Features**:
- MD5 hash-based fingerprinting
- Baseline comparison
- New/removed item detection
- Change history tracking (last 100 events)
- Configurable key fields

**Example Projects Using This**:
- NYPD Procurement Tracker (detects new opportunities)

### Pattern 3: Email Alerts
**File**: `patterns/email_patterns.py`
**Use When**: Need to notify users of changes or events
**Features**:
- SMTP integration (Gmail, Outlook, custom)
- HTML and plain-text email templates
- Environment variable configuration
- Attachment support
- Professional formatting

**Example Projects Using This**:
- NYPD Procurement Tracker (alerts on new opportunities)

### Pattern 4: Flask Dashboard
**File**: `templates/web-dashboard/`
**Use When**: Need real-time web interface for monitoring
**Features**:
- Flask backend with REST APIs
- Chart.js for visualization
- Responsive design (mobile-friendly)
- Real-time data updates
- Search/filter functionality
- Auto-refresh capability

**Example Projects Using This**:
- NYPD Procurement Tracker (monitoring dashboard)
- Application Rationalization Tool (analysis dashboard)

### Pattern 5: Task Scheduling
**File**: `patterns/scheduler_patterns.py`
**Use When**: Need to run tasks automatically at intervals
**Features**:
- Python schedule library integration
- Configurable intervals
- Background operation
- Error handling and retries
- Logging and monitoring

**Example Projects Using This**:
- NYPD Procurement Tracker (hourly scraping)

### Pattern 6: Data Analysis
**File**: `patterns/analysis_patterns.py`
**Use When**: Need to process, analyze, or transform data
**Features**:
- Pandas DataFrame operations
- Excel/CSV processing
- Data validation and cleaning
- Statistical analysis
- Aggregation and grouping

**Example Projects Using This**:
- Application Rationalization Tool (app portfolio analysis)
- Capital Projects Planner (project data analysis)

### Pattern 7: Documentation Generation
**File**: `patterns/documentation_patterns.py`
**Use When**: Need to create professional documentation
**Features**:
- Markdown generation
- Word document creation (python-docx)
- Problem/solution statements
- One-pagers
- Executive summaries
- ROI calculations

**Example Projects Using This**:
- Application Rationalization Tool (generates comprehensive docs)
- Capital Projects Planner (project documentation)

---

## Scoring & Risk Patterns Library

These advanced patterns enable multi-component scoring, risk classification, alert generation, and KPI benchmarking. Extracted from production systems like Contract Oversight System.

### Pattern 8: Weighted Scoring Engine
**File**: `patterns/scoring/weighted_scoring.py`
**Use When**: Need to calculate health scores, performance ratings, or risk assessments
**Features**:
- Multi-component weighted scoring (configurable weights)
- Automatic grade assignment (A-F)
- Risk level determination (Critical/High/Medium/Low)
- Batch scoring for multiple entities
- Factory functions for common use cases (contracts, employees, vendors)
- Score normalization (handles different scales)

**Key Classes**:
- `WeightedScoringEngine` - Main scoring engine
- `AggregatedScoringEngine` - Aggregate scores across entities (e.g., vendor from contracts)
- `ScoreComponent` - Define individual scoring dimensions

**Example Usage**:
```python
from patterns.scoring import WeightedScoringEngine, ScoreComponent, ScoreDirection

components = [
    ScoreComponent("cost_variance", weight=0.30, direction=ScoreDirection.LOWER_IS_BETTER,
                  min_value=0, max_value=50),
    ScoreComponent("performance", weight=0.35, direction=ScoreDirection.HIGHER_IS_BETTER,
                  min_value=0, max_value=100),
    ScoreComponent("compliance", weight=0.35, direction=ScoreDirection.HIGHER_IS_BETTER,
                  min_value=0, max_value=100),
]

engine = WeightedScoringEngine(components)
result = engine.score({"cost_variance": 15, "performance": 85, "compliance": 95})
print(f"Score: {result.overall_score}, Grade: {result.grade}, Risk: {result.risk_level}")
```

### Pattern 9: Alert Rules Engine
**File**: `patterns/scoring/alert_rules_engine.py`
**Use When**: Need rule-based alert generation for monitoring dashboards
**Features**:
- Declarative rule definitions (ID, condition, severity, message)
- Multiple severity levels (Critical, High, Medium, Low, Info)
- Category-based organization (Cost, Schedule, Compliance, etc.)
- Automatic alert message formatting with entity data
- Statistics and distribution tracking
- Pre-built rule sets for common domains

**Key Classes**:
- `AlertRulesEngine` - Main alert engine
- `AlertRule` - Define individual alert rules
- `Alert` - Generated alert objects
- `AlertSeverity` - Severity enumeration

**Example Usage**:
```python
from patterns.scoring import AlertRulesEngine, AlertRule, AlertSeverity, AlertCategory

engine = AlertRulesEngine()
engine.add_rule(AlertRule(
    rule_id="budget_warning",
    name="Budget Warning",
    severity=AlertSeverity.HIGH,
    category=AlertCategory.COST,
    condition=lambda e: e.get("budget_variance", 0) > 10,
    message_template="Budget exceeded by {budget_variance}%"
))

alerts = engine.evaluate_all(entities)
```

### Pattern 10: Risk Classification
**File**: `patterns/scoring/risk_classification.py`
**Use When**: Need to convert scores to risk levels with thresholds
**Features**:
- Configurable risk thresholds
- Supports both directions (higher_is_riskier, lower_is_riskier)
- Color and icon mappings for visualization
- Multi-dimensional risk classification
- Action-required recommendations per level
- Batch classification

**Key Classes**:
- `RiskClassifier` - Single-dimension risk classification
- `MultiDimensionalRiskClassifier` - Multi-factor risk aggregation
- `RiskLevel` - Risk level enumeration with colors/icons
- `RiskThreshold` - Threshold configuration

**Example Usage**:
```python
from patterns.scoring import create_health_score_classifier

classifier = create_health_score_classifier()  # For 0-100 health scores
result = classifier.classify(score=45, entity_id="PROJECT-001")
print(f"Risk: {result.level.icon} {result.level.value}")  # 🟠 High
print(f"Action: {result.action_required}")
```

### Pattern 11: KPI Benchmarking
**File**: `patterns/scoring/benchmark_engine.py`
**Use When**: Need to compare performance against industry benchmarks
**Features**:
- KPI definitions with benchmarks and direction
- Gap analysis (actual vs benchmark)
- Score calculation with direction awareness
- Category aggregation with weighting
- Automatic recommendation generation
- Pre-built benchmark sets (Procurement, HR, Manufacturing)
- Peer comparison and percentile ranking

**Key Classes**:
- `BenchmarkEngine` - Main benchmarking engine
- `KPIDefinition` - Define KPIs with benchmarks
- `BenchmarkReport` - Comprehensive analysis report
- `CategoryScore` - Category-level aggregation

**Example Usage**:
```python
from patterns.scoring import create_procurement_benchmarks

engine = create_procurement_benchmarks()
report = engine.analyze({
    "spend_under_management": 72,
    "contract_coverage": 68,
    "cost_savings": 14,
    "cycle_time": 52,
})
print(f"Grade: {report.grade}, Score: {report.overall_score}")
for rec in report.recommendations:
    print(f"  > {rec}")
```

### Using the Complete Scoring Suite
Use pattern `scoring-suite` in create_project.py to get all four patterns together:
```bash
python create_project.py --name "Contract Monitor" --patterns dashboard,scoring-suite
```

---

## ML/AI Patterns Library

Advanced machine learning patterns for forecasting, ensemble prediction, and model management. Extracted from production systems like Stock Analysis.

### Pattern 12: Feature Engineering Pipeline
**File**: `patterns/ml/feature_pipeline.py`
**Use When**: Need to generate technical indicators and features for ML models
**Features**:
- 50+ technical indicators (RSI, MACD, Bollinger Bands, etc.)
- Momentum, volatility, trend, and volume features
- Statistical features (skewness, kurtosis, z-scores)
- Cyclical encoding for time features (sin/cos)
- Lag features for time series modeling
- Custom feature registration

**Key Classes**:
- `FeaturePipeline` - Main feature generator
- `FeatureCategory` - Organize features by type
- `FeatureDefinition` - Define custom features

**Example Usage**:
```python
from patterns.ml import FeaturePipeline, FeatureCategory

# Create pipeline with selected categories
pipeline = FeaturePipeline(categories=[
    FeatureCategory.PRICE,
    FeatureCategory.MOMENTUM,
    FeatureCategory.VOLATILITY
])

# Generate features from OHLCV data
features_df = pipeline.generate_features(df)

# Or use factory function for technical analysis
from patterns.ml import create_technical_analysis_pipeline
pipeline = create_technical_analysis_pipeline()
```

### Pattern 13: Time Series Forecasting
**File**: `patterns/ml/time_series_models.py`
**Use When**: Need to forecast future values from historical data
**Features**:
- Multiple forecasting methods (Moving Average, Exponential Smoothing, ARIMA)
- Ensemble forecasting combining multiple methods
- Confidence intervals and uncertainty quantification
- Automatic seasonality detection
- Holt-Winters for seasonal data

**Key Classes**:
- `TimeSeriesForecaster` - Unified forecasting interface
- `ForecastMethod` - Available methods (ARIMA, ENSEMBLE, etc.)
- `ForecastResult` - Predictions with confidence bounds
- `SeasonalityType` - Seasonality patterns

**Example Usage**:
```python
from patterns.ml import TimeSeriesForecaster, ForecastMethod

# Create ensemble forecaster
forecaster = TimeSeriesForecaster(method=ForecastMethod.ENSEMBLE)
forecaster.fit(df['close'])

# Generate forecast with confidence intervals
result = forecaster.predict_with_intervals(horizon=30)
print(f"Predictions: {result.predictions}")
print(f"Confidence: {result.confidence_level}")

# Or use factory functions
from patterns.ml import create_auto_forecaster
forecaster = create_auto_forecaster()  # Auto seasonality detection
```

### Pattern 14: Ensemble Predictor
**File**: `patterns/ml/ensemble_predictor.py`
**Use When**: Need to combine multiple models for better predictions
**Features**:
- Multiple ensemble methods (Weighted Average, Stacking, Voting)
- Uncertainty quantification from model disagreement
- Auto-weighting based on validation error
- Meta-learner training for stacking
- Model enable/disable for experimentation

**Key Classes**:
- `EnsemblePredictor` - Multi-model orchestrator
- `EnsembleMethod` - Combination methods
- `PredictionResult` - Prediction with uncertainty
- `ModelWrapper` - Standardize model interfaces

**Example Usage**:
```python
from patterns.ml import EnsemblePredictor, EnsembleMethod

# Create weighted ensemble
ensemble = EnsemblePredictor(method=EnsembleMethod.WEIGHTED_AVERAGE)

# Register models with weights
ensemble.register_model('model_a', model_a, weight=0.3)
ensemble.register_model('model_b', model_b, weight=0.5)
ensemble.register_model('model_c', model_c, weight=0.2)

# Get prediction with uncertainty
result = ensemble.predict_with_uncertainty(X_test)
print(f"Prediction: {result.prediction}")
print(f"Confidence: {result.confidence}")
```

### Pattern 15: Model Registry
**File**: `patterns/ml/model_registry.py`
**Use When**: Need model versioning, storage, and lifecycle management
**Features**:
- Model versioning with automatic version numbers
- Lifecycle stages (Development, Staging, Production, Archived)
- Model comparison with metrics
- Tag-based search
- Export/import models
- Checksum integrity verification

**Key Classes**:
- `ModelRegistry` - Main registry interface
- `ModelVersion` - Model with metadata
- `ModelStage` - Lifecycle stages
- `ModelMetadata` - Version information

**Example Usage**:
```python
from patterns.ml import ModelRegistry, ModelStage

# Initialize registry
registry = ModelRegistry(storage_path='models/')

# Register a model
version = registry.register(
    name='price_predictor',
    model=trained_model,
    metrics={'rmse': 0.05, 'mae': 0.03},
    tags=['production', 'v2']
)

# Promote to production
registry.transition_stage('price_predictor', version.version, ModelStage.PRODUCTION)

# Load production model
model = registry.load('price_predictor', stage=ModelStage.PRODUCTION)

# Compare versions
comparison = registry.compare_versions('price_predictor')
```

### Using the Complete ML Suite
Use pattern `ml-suite` in create_project.py to get all four ML patterns together:
```bash
python create_project.py --name "Demand Forecaster" --patterns dashboard,ml-suite
```

---

## Razors (Analysis Patterns) Library

Reusable analysis and filtering patterns extracted from procurement, bid management, and risk analysis tools.

### Pattern 16: Temporal Decay Scoring
**File**: `patterns/razors/temporal_decay.py`
**Use When**: Need time-based scoring where freshness matters
**Features**:
- Bracket-based scoring (7 days = 100, 14 days = 90, etc.)
- Multiple decay functions (linear, exponential, logarithmic, sigmoid)
- Expiration tracking with urgency levels
- Freshness labels (Fresh, Recent, Aging, Stale)

**Key Classes**:
- `TemporalDecay` - Main decay scorer
- `ExpirationTracker` - Track expiring items with alerts
- `DecayFunction` - Decay function types

**Example Usage**:
```python
from patterns.razors import TemporalDecay, create_recency_scorer

# Create recency scorer with brackets
decay = create_recency_scorer()
score = decay.score(days_old=25)  # Returns 75

# Or with date
result = decay.score_with_details(date=article_date)
print(f"Score: {result.score}, Freshness: {result.freshness_label}")

# Expiration tracking
from patterns.razors import ExpirationTracker
tracker = ExpirationTracker(warning_days=90, critical_days=30)
status = tracker.check(contract_end_date)
print(f"Level: {status['level']}, Action Required: {status['action_required']}")
```

### Pattern 17: Filter Chains
**File**: `patterns/razors/filter_chains.py`
**Use When**: Need composable, complex filtering logic
**Features**:
- Fluent API for building filters
- AND/OR logic between filter groups
- Negative filters (exclusion lists)
- Threshold-based priority classification
- Keyword filtering (positive and negative)

**Key Classes**:
- `FilterChain` - Composable filter builder
- `NegativeKeywordFilter` - Exclude by keywords
- `ThresholdFilter` - Score-based priority classification

**Example Usage**:
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

# Negative keyword filtering (like procurement)
from patterns.razors import create_procurement_keyword_filter
kw_filter = create_procurement_keyword_filter()
filtered = kw_filter.apply(rfps, fields=['title', 'description'])
```

### Pattern 18: Deduplication & Change Detection
**File**: `patterns/razors/deduplication.py`
**Use When**: Need to deduplicate data or track changes over time
**Features**:
- Multi-key deduplication (primary + fallback keys)
- Change detection between datasets (new, modified, removed)
- Fingerprint/hash generation for versioning
- Configurable matching strategies

**Key Classes**:
- `Deduplicator` - Multi-key deduplication
- `ChangeDetector` - Track changes between datasets
- `FingerprintGenerator` - Hash-based change detection

**Example Usage**:
```python
from patterns.razors import create_procurement_deduplicator, ChangeDetector

# Deduplication with fallback keys
dedup = create_procurement_deduplicator()  # Uses agency+bid_number OR agency+title
unique_bids = dedup.deduplicate(all_bids)

# Change detection
detector = ChangeDetector(
    key_fields=['contract_id'],
    compare_fields=['status', 'amount', 'end_date']
)
changes = detector.detect_changes(new_data, existing_data)
print(f"New: {changes['new_count']}, Modified: {changes['modified_count']}")

# Fingerprinting
from patterns.razors import FingerprintGenerator
fp = FingerprintGenerator(exclude_fields=['updated_at'])
if not fp.compare(old_doc, new_doc):
    print("Document changed!")
```

### Using the Complete Razors Suite
Use pattern `razors-suite` in create_project.py to get all three patterns:
```bash
python create_project.py --name "Lead Tracker" --patterns dashboard,razors-suite
```

---

## Export & Reporting Patterns Library

Professional document and report generation patterns extracted from application-rationalization-tool.

### Pattern 19: Report Generator
**File**: `patterns/export/report_generator.py`
**Use When**: Need multi-format data exports and reports
**Features**:
- Multiple formats: CSV, JSON, Excel, HTML, Markdown
- Structured sections (metrics, tables, lists, summaries)
- Professional Excel formatting with styled headers
- HTML reports with embedded CSS styling
- Batch export to all formats at once

**Key Classes**:
- `ReportGenerator` - Multi-section report builder
- `DataExporter` - Simple data export utility
- `ReportSection` - Individual report sections

**Example Usage**:
```python
from patterns.export import ReportGenerator, ReportFormat

# Create structured report
report = ReportGenerator(title="Q4 Performance Report")
report.add_metrics("Key Metrics", {
    "Total Revenue": "$1.2M",
    "Growth Rate": "15%",
    "Active Customers": "5,000"
})
report.add_table("Sales by Region", sales_data)
report.add_list("Key Achievements", achievements)

# Export to multiple formats
report.export("report.xlsx", ReportFormat.EXCEL)
report.export("report.html", ReportFormat.HTML)
report.export_all("report")  # Creates all formats

# Quick data export
from patterns.export import DataExporter
exporter = DataExporter()
exporter.to_excel(data, "output.xlsx")
exporter.to_csv(data, "output.csv")
```

### Pattern 20: Document Builder
**File**: `patterns/export/document_builder.py`
**Use When**: Need professional Word documents or one-pagers
**Features**:
- Word document generation (python-docx)
- Multiple style presets (Corporate, Executive, Technical)
- Tables, lists, images, quotes
- One-pager builder with tight formatting
- Fallback to Markdown/HTML if docx unavailable

**Key Classes**:
- `DocumentBuilder` - Word document builder
- `OnePagerBuilder` - Compact one-page documents
- `DocumentStyle` - Style presets

**Example Usage**:
```python
from patterns.export import DocumentBuilder, DocumentStyle

# Create Word document
doc = DocumentBuilder(title="Project Proposal", style=DocumentStyle.CORPORATE)
doc.add_heading("Executive Summary", level=1)
doc.add_paragraph("This proposal outlines our approach to...")
doc.add_heading("Key Benefits", level=2)
doc.add_bullet_list([
    "Reduce costs by 25%",
    "Improve efficiency by 40%",
    "Faster time to market"
])
doc.add_table(timeline_data, headers=['Phase', 'Duration', 'Deliverables'])
doc.save("proposal.docx")

# One-pager
from patterns.export import OnePagerBuilder
one_pager = OnePagerBuilder(title="Product Overview")
one_pager.add_paragraph("Brief introduction...")
one_pager.add_key_value_table({"Price": "$99/mo", "Users": "Unlimited"})
one_pager.save("overview.docx")

# Factory function
from patterns.export import create_proposal_document
doc = create_proposal_document(
    title="Service Proposal",
    executive_summary="...",
    problem_statement="...",
    proposed_solution="...",
    benefits=["Benefit 1", "Benefit 2"]
)
doc.save("proposal.docx")
```

### Using the Complete Export Suite
Use pattern `export-suite` in create_project.py to get both patterns:
```bash
python create_project.py --name "Report Tool" --patterns dashboard,export-suite
```

---

## Notifications Patterns Library

Alert management, email digests, and multi-channel notification routing.

### Pattern 21: Email Digest Builder
**File**: `patterns/notifications/email_digest.py`
**Use When**: Need professional HTML email summaries and alerts
**Features**:
- Multiple theme presets (Professional, Alert, Success, Minimal, Dark)
- Summary cards with metrics
- Alert sections with severity styling
- Data tables and lists
- Action buttons with links
- Plain-text fallback generation

**Key Classes**:
- `EmailDigestBuilder` - Main digest builder
- `DigestTheme` - Theme presets
- `AlertEmailBuilder` - Single-alert quick builder

**Example Usage**:
```python
from patterns.notifications import EmailDigestBuilder, DigestTheme

# Create daily digest
digest = EmailDigestBuilder("Daily Summary", theme=DigestTheme.PROFESSIONAL)
digest.add_summary({"New Alerts": 12, "Resolved": 5})
digest.add_table("Recent Items", items_data)
digest.add_action_button("View Dashboard", "https://app.example.com")

html, text = digest.build()
```

### Pattern 22: Alert Aggregator
**File**: `patterns/notifications/alert_aggregator.py`
**Use When**: Need to deduplicate, rate-limit, and group alerts
**Features**:
- Deduplication within configurable time window
- Rate limiting per priority level
- Quiet hours suppression (overnight)
- Alert grouping by category/source
- Priority escalation
- Expiration handling

**Key Classes**:
- `AlertAggregator` - Main aggregator with rate limiting
- `AlertPriority` - Priority levels (Critical, High, Medium, Low)
- `AlertBuffer` - Buffer for batch processing

**Example Usage**:
```python
from patterns.notifications import AlertAggregator, AlertPriority

aggregator = AlertAggregator(
    dedup_window_minutes=60,
    max_alerts_per_hour=20,
    quiet_hours=(22, 7)
)

# Add alerts (duplicates are merged)
aggregator.add_alert("Server Down", "Web server not responding", AlertPriority.CRITICAL)

# Get ready alerts
for alert in aggregator.get_pending_alerts():
    send_notification(alert)
    aggregator.mark_sent(alert.id)
```

### Pattern 23: Notification Router
**File**: `patterns/notifications/notification_router.py`
**Use When**: Need to route alerts to multiple channels (Email, Slack, Teams)
**Features**:
- Multiple channel support (Email, Slack, Teams, Webhook)
- Routing rules with conditions
- Fallback channels
- Retry logic
- Delivery tracking

**Key Classes**:
- `NotificationRouter` - Route to multiple channels
- `SlackHandler`, `TeamsHandler`, `WebhookHandler` - Channel handlers
- `RoutingRule` - Conditional routing

**Example Usage**:
```python
from patterns.notifications import NotificationRouter, SlackHandler, Channel

router = NotificationRouter()
router.add_channel(Channel.SLACK, SlackHandler(webhook_url))
router.set_default_channels([Channel.SLACK])

# Route notification
from patterns.notifications import Notification, NotificationPriority
notification = Notification(
    title="Alert",
    message="System issue detected",
    priority=NotificationPriority.HIGH
)
results = router.send(notification)
```

### Using the Complete Notifications Suite
```bash
python create_project.py --name "Alert System" --patterns dashboard,notifications-suite
```

---

## Connectors Patterns Library

Data connection and fetching patterns for APIs, web scraping, and data sources.

### Pattern 24: API Client
**File**: `patterns/connectors/api_client.py`
**Use When**: Need robust API access with rate limiting and retry
**Features**:
- Multiple auth methods (Bearer, API Key, Basic, OAuth)
- Token bucket rate limiting
- Automatic retry with exponential backoff
- Response caching
- Pagination support
- Request/response hooks

**Key Classes**:
- `APIClient` - Main client with all features
- `APIClientBuilder` - Fluent builder pattern
- `RateLimiter` - Token bucket rate limiter
- `BearerAuth`, `APIKeyAuth`, `BasicAuth` - Auth handlers

**Example Usage**:
```python
from patterns.connectors import APIClientBuilder

client = (APIClientBuilder("https://api.example.com")
    .with_bearer_token("token")
    .with_rate_limit(10)  # 10 requests/second
    .with_retry(max_retries=3)
    .with_cache(ttl=300)
    .build())

# Simple requests
users = client.get("/users")
result = client.post("/users", json={"name": "John"})

# Paginated
all_items = client.get_all_pages("/items", page_param="page")
```

### Pattern 25: Scraper Base
**File**: `patterns/connectors/scraper_base.py`
**Use When**: Need robust web scraping with anti-detection
**Features**:
- Session management with cookies
- User agent rotation
- Random delay variance
- Proxy support and rotation
- Retry with backoff
- BeautifulSoup integration

**Key Classes**:
- `ScraperBase` - Main scraper with all features
- `ScraperSession` - Stateful session with cookies
- `PageScraper` - Pagination and link following

**Example Usage**:
```python
from patterns.connectors import ScraperBase, ScraperConfig

scraper = ScraperBase(ScraperConfig(
    rate_limit_seconds=2.0,
    rotate_user_agents=True,
    random_delay=True
))

# Simple fetch
result = scraper.get("https://example.com")
soup = scraper.get_soup("https://example.com")

# Session-based (maintains cookies)
with scraper.session() as session:
    session.get("https://example.com/login")
    session.post("https://example.com/login", data={"user": "x"})
    data = session.get("https://example.com/protected")
```

### Pattern 26: Data Source Registry
**File**: `patterns/connectors/data_source_registry.py`
**Use When**: Need unified access to multiple data sources
**Features**:
- Register multiple sources (API, File, Memory)
- Health monitoring
- Access tracking
- Caching wrapper
- Fallback sources

**Key Classes**:
- `DataSourceRegistry` - Main registry
- `APISource`, `FileSource`, `MemorySource` - Source types
- `CachedSource`, `FallbackSource` - Wrappers

**Example Usage**:
```python
from patterns.connectors import DataSourceRegistry, APISource, FileSource

registry = DataSourceRegistry()
registry.register("users_api", APISource(base_url="https://api.example.com"))
registry.register("backup", FileSource("data/backup.json"))

# Fetch from source
result = registry.fetch("users_api", "/users")

# Health check all
health = registry.health_check_all()
```

### Using the Complete Connectors Suite
```bash
python create_project.py --name "Data Pipeline" --patterns connectors-suite,scheduling-suite
```

---

## Scheduling Patterns Library

Job scheduling, pipeline orchestration, and checkpoint/resume functionality.

### Pattern 27: Job Scheduler
**File**: `patterns/scheduling/job_scheduler.py`
**Use When**: Need cron-like job scheduling
**Features**:
- Interval, daily, hourly, cron scheduling
- Job dependencies
- Retry with backoff
- Execution history
- Concurrent job limits

**Key Classes**:
- `JobScheduler` - Main scheduler
- `Job` - Job definition
- `Schedule` - Schedule configuration
- `SimpleScheduler` - Decorator-based scheduler

**Example Usage**:
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

scheduler.start()
```

### Pattern 28: Pipeline Orchestrator
**File**: `patterns/scheduling/pipeline_orchestrator.py`
**Use When**: Need DAG-based task execution with dependencies
**Features**:
- Dependency-based execution order
- Parallel execution of independent tasks
- Retry logic per task
- Shared context between tasks
- Visual DAG representation

**Key Classes**:
- `Pipeline` - DAG-based pipeline
- `Task` - Task definition
- `PipelineBuilder` - Fluent builder
- `ETLPipeline` - Specialized for ETL

**Example Usage**:
```python
from patterns.scheduling import Pipeline, Task

pipeline = Pipeline("data_pipeline", max_parallel=4)

pipeline.add_task(Task("extract", extract_func))
pipeline.add_task(Task("transform", transform_func, depends_on=["extract"]))
pipeline.add_task(Task("load", load_func, depends_on=["transform"]))

result = pipeline.run()
print(f"Status: {result.status}, Duration: {result.duration_ms}ms")
```

### Pattern 29: Checkpoint Manager
**File**: `patterns/scheduling/checkpoint_manager.py`
**Use When**: Need to resume long-running processes from failure
**Features**:
- Save and load state
- Automatic versioning
- Progress tracking
- Checkpoint history
- Incremental processing

**Key Classes**:
- `CheckpointManager` - Main checkpoint manager
- `IncrementalProcessor` - Process items with auto-checkpoint

**Example Usage**:
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

### Using the Complete Scheduling Suite
```bash
python create_project.py --name "ETL Pipeline" --patterns scheduling-suite,connectors-suite
```

---

## How to Use This Framework

### Method 1: Quick Start with Prompts

When you have a new business idea, use these prompts with Claude Code:

**For Web Scraping + Monitoring:**
```
"Using the universal-business-solution-framework, build a tool that:
- Scrapes [URL] for [data type]
- Detects when new [items] are added
- Sends email alerts
- Shows a dashboard

Use patterns: scraping + change-detection + email + dashboard"
```

**For Data Analysis:**
```
"Using the universal-business-solution-framework, build a tool that:
- Processes [data source]
- Analyzes [metrics]
- Generates reports

Use patterns: data-analysis + documentation"
```

**For Automation:**
```
"Using the universal-business-solution-framework, build a tool that:
- Runs every [interval]
- Performs [task]
- Notifies on [condition]

Use patterns: scheduler + notification"
```

### Method 2: Use the Project Generator

```bash
cd C:\Users\dada_\OneDrive\Documents\universal-business-solution-framework
python create_project.py --name "My New Tool" --patterns scraping,dashboard,email
```

### Method 3: Manual Pattern Assembly

1. Copy pattern files you need from `patterns/`
2. Copy template structure from `templates/`
3. Customize for your specific use case
4. Reference example projects in `examples/`

---

## Available Pattern Combinations

### Combination 1: Data Monitoring System
**Patterns**: Scraping + Change Detection + Email + Dashboard
**Best For**: Tracking websites for changes (procurement, job postings, price tracking)
**Time to Build**: 2-4 hours
**Example**: NYPD Procurement Tracker

### Combination 2: Business Analysis Tool
**Patterns**: Data Analysis + Documentation + Dashboard
**Best For**: Portfolio analysis, cost optimization, strategic planning
**Time to Build**: 4-6 hours
**Example**: Application Rationalization Tool

### Combination 3: Automated Data Pipeline
**Patterns**: Scraping/API + Data Processing + Scheduler + Storage
**Best For**: ETL processes, data aggregation, regular reporting
**Time to Build**: 3-5 hours

### Combination 4: Alert & Notification System
**Patterns**: Data Source + Change Detection + Email/Slack
**Best For**: Monitoring for specific events, compliance tracking
**Time to Build**: 2-3 hours

### Combination 5: Executive Dashboard
**Patterns**: Data Source + Analysis + Dashboard + Documentation
**Best For**: KPI tracking, performance monitoring, executive reporting
**Time to Build**: 4-6 hours

---

## Project Structure Template

Every project created from this framework follows this structure:

```
project-name/
├── src/
│   ├── scraper.py              # Data collection (if using scraping pattern)
│   ├── analyzer.py             # Data processing (if using analysis pattern)
│   ├── change_detector.py      # Change detection (if using that pattern)
│   └── scheduler.py            # Automation (if using scheduler pattern)
├── web/                        # Dashboard (if using dashboard pattern)
│   ├── app.py                  # Flask application
│   ├── templates/
│   │   └── index.html
│   └── static/
│       ├── css/
│       └── js/
├── data/
│   ├── baseline.json           # Current state
│   └── history.json            # Change history
├── output/                     # Exports and reports
├── docs/                       # Documentation
├── requirements.txt            # Python dependencies
├── README.md                   # Project overview
├── SETUP_GUIDE.md             # Setup instructions
└── .env.example               # Environment variables template
```

---

## Example Projects (Reference Implementations)

### 1. NYPD Procurement Tracker
**Location**: `C:\Users\dada_\OneDrive\Documents\nypd-procurement-tracker\`
**Patterns Used**: Scraping + Change Detection + Email + Dashboard + Scheduler
**Business Problem**: Be first to respond to new procurement opportunities
**Value**: Competitive advantage, first-mover benefit
**Key Features**:
- Scrapes NYPD M/WBE page for opportunities
- MD5 hash-based change detection
- Instant email alerts with HTML formatting
- Real-time web dashboard with Chart.js
- Auto-refresh every 5 minutes
- 76 opportunities tracked

### 2. Application Rationalization Tool
**Location**: `C:\Users\dada_\OneDrive\Documents\application-rationalization-tool\`
**Patterns Used**: Data Analysis + Documentation + Dashboard
**Business Problem**: IT portfolio bloat, redundant applications, high costs
**Value**: 23x ROI in Year 1
**Key Features**:
- Analyzes application portfolio
- Identifies redundancies and consolidation opportunities
- Generates comprehensive documentation
- Problem/solution statements
- One-pagers and pitch decks

### 3. Capital Projects Planner
**Location**: `C:\Users\dada_\OneDrive\Documents\application-rationalization-tool\capital_projects\`
**Patterns Used**: Data Processing + Analysis + Documentation
**Business Problem**: Infrastructure delivery delays, budget overruns
**Value**: 44x ROI in Year 1
**Key Features**:
- Project portfolio analysis
- Resource allocation optimization
- Executive documentation generation

---

## Technology Stack

### Backend
- **Python 3.8+** - Primary language
- **Flask 3.0+** - Web framework for dashboards
- **BeautifulSoup4** - Web scraping
- **Pandas** - Data analysis
- **Requests** - HTTP client
- **Schedule** - Task scheduling
- **python-dotenv** - Environment configuration
- **python-docx** - Word document generation

### Frontend (Dashboard Pattern)
- **HTML5/CSS3** - Structure and styling
- **JavaScript (ES6+)** - Interactive functionality
- **Chart.js** - Data visualization
- **Fetch API** - Async data loading

### Storage
- **JSON** - Lightweight data persistence
- **CSV** - Data exports
- **SQLite** (optional) - Relational data
- **PostgreSQL** (optional) - Production databases

### Notifications
- **SMTP** - Email alerts
- **Twilio** (optional) - SMS
- **Slack API** (optional) - Team notifications

---

## Best Practices

### 1. Always Start with Business Value
- What problem does this solve?
- What's the ROI or competitive advantage?
- Who are the users?

### 2. Use Modular Design
- Each pattern is independent
- Easy to add/remove features
- Test components separately

### 3. Document Everything
- README for overview
- SETUP_GUIDE for installation
- Inline code comments
- API documentation

### 4. Error Handling
- Graceful degradation
- Logging and monitoring
- Retry logic for external services
- User-friendly error messages

### 5. Security
- Environment variables for secrets
- Never commit credentials
- Validate all inputs
- Use HTTPS for production

---

## Quick Reference Commands

### Create New Project
```bash
python create_project.py --name "Project Name" --patterns scraping,dashboard
```

### Common Pattern Combinations
- `scraping,change-detection,email` - Monitoring system
- `analysis,documentation,dashboard` - Business analysis tool
- `scraping,scheduler,storage` - Data pipeline
- `api,dashboard,notifications` - Real-time monitoring

### Install Dependencies
```bash
cd project-folder
pip install -r requirements.txt
```

### Run Dashboard
```bash
python web/app.py
# Open http://localhost:5001
```

### Run Scheduled Task
```bash
python src/scheduler.py --interval 60
```

---

## Extension Points

The framework is designed to be extended. Add new patterns for:

### Additional Data Sources
- REST APIs
- GraphQL
- Databases (SQL, NoSQL)
- Cloud storage (S3, Azure Blob)
- File uploads

### Additional Notifications
- SMS (Twilio)
- Push notifications
- Slack/Teams webhooks
- Mobile apps

### Additional Visualizations
- D3.js for complex charts
- Maps (Leaflet, Mapbox)
- Real-time graphs
- Export to PowerBI/Tableau

### Additional Automation
- Docker containerization
- CI/CD pipelines
- Cloud deployment (AWS, Azure, GCP)
- Kubernetes orchestration

---

## Support & Troubleshooting

### Common Issues

**Import errors**: Ensure requirements.txt is installed
**Port conflicts**: Change port in web/app.py
**Email not sending**: Check SMTP settings in .env
**Scraping fails**: Website structure may have changed

### Getting Help

1. Check example projects for reference
2. Review pattern documentation
3. Check pattern files for inline comments
4. Review framework examples

---

## Framework Maintenance

### Adding New Patterns
1. Create pattern file in `patterns/`
2. Document in this file
3. Add example usage
4. Update `create_project.py`

### Updating Existing Patterns
1. Update pattern file
2. Update documentation
3. Test with example projects
4. Update version number

---

## Version History

**v1.4** - Infrastructure Patterns (Current)
- Added 3 new Notification patterns:
  - Email Digest Builder (HTML templates, themes, sections)
  - Alert Aggregator (dedup, rate-limit, quiet hours)
  - Notification Router (Slack, Teams, Webhook, Email)
- Added 3 new Connector patterns:
  - API Client (rate limiting, retry, caching, auth)
  - Scraper Base (sessions, anti-detection, proxies)
  - Data Source Registry (unified access, health monitoring)
- Added 3 new Scheduling patterns:
  - Job Scheduler (cron, dependencies, retry)
  - Pipeline Orchestrator (DAG, parallel execution)
  - Checkpoint Manager (resume, incremental processing)
- Total: 29 patterns across 9 categories

**v1.3** - Razors & Export Patterns
- Added 3 new Razor patterns:
  - Temporal Decay Scoring (recency, expiration tracking)
  - Filter Chains (composable filters, negative keywords, thresholds)
  - Deduplication & Change Detection (multi-key, fingerprinting)
- Added 2 new Export patterns:
  - Report Generator (CSV, JSON, Excel, HTML, Markdown)
  - Document Builder (Word documents with styles)
- Patterns extracted from Procurement Intel, Bid Management, App Rationalization
- Total: 20 patterns across 6 categories

**v1.2** - ML/AI Patterns
- Added 4 new ML patterns:
  - Feature Engineering Pipeline (50+ technical indicators)
  - Time Series Forecasting (ARIMA, Ensemble, Holt-Winters)
  - Ensemble Predictor (Weighted, Stacking, Voting)
  - Model Registry (versioning, lifecycle management)
- Patterns extracted from Stock Analysis project
- Zero external dependencies (uses only pandas/numpy)
- Factory functions for common ML workflows

**v1.1** - Scoring & Risk Patterns
- Added 4 new scoring patterns:
  - Weighted Scoring Engine
  - Alert Rules Engine
  - Risk Classification
  - KPI Benchmarking
- Patterns extracted from Contract Oversight System
- Updated create_project.py with pattern categories
- Factory functions for common use cases

**v1.0** - Initial framework creation
- 7 core patterns
- 3 example projects
- Project generator
- Complete documentation

---

**This framework represents proven, production-ready patterns extracted from real business solutions. Use it to rapidly prototype and deploy new business automation tools.**
