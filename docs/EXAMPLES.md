# Usage Examples

Real-world examples demonstrating how to use the Universal Business Solution Framework.

---

## Table of Contents

1. [Website Change Monitor](#1-website-change-monitor)
2. [Multi-Source Data Pipeline](#2-multi-source-data-pipeline)
3. [Alert Management System](#3-alert-management-system)
4. [Business Scoring Dashboard](#4-business-scoring-dashboard)
5. [ML Forecasting Pipeline](#5-ml-forecasting-pipeline)
6. [Document Generation System](#6-document-generation-system)

---

## 1. Website Change Monitor

Monitor a website for new content and send notifications.

### Setup
```bash
python create_project.py --name "Site Monitor" \
  --patterns scraping,change-detection,notifications-suite,scheduler
```

### Code Example

```python
from patterns.connectors import ScraperBase, ScraperConfig
from patterns.razors import Deduplicator, ChangeDetector
from patterns.notifications import (
    AlertAggregator, AlertPriority,
    EmailDigestBuilder, DigestTheme,
    create_slack_router
)
from patterns.scheduling import JobScheduler, Job, Schedule
import json
from pathlib import Path

# Configuration
SITE_URL = "https://example.com/listings"
DATA_FILE = "data/listings.json"
SLACK_WEBHOOK = "https://hooks.slack.com/services/..."

# Initialize components
scraper = ScraperBase(ScraperConfig(
    rate_limit_seconds=2.0,
    rotate_user_agents=True
))

aggregator = AlertAggregator(
    dedup_window_minutes=60,
    max_alerts_per_hour=10
)

router = create_slack_router(SLACK_WEBHOOK)

def load_existing():
    """Load previously seen items."""
    path = Path(DATA_FILE)
    if path.exists():
        return json.loads(path.read_text())
    return []

def save_items(items):
    """Save items to file."""
    Path(DATA_FILE).parent.mkdir(exist_ok=True)
    Path(DATA_FILE).write_text(json.dumps(items, indent=2))

def check_for_updates():
    """Main monitoring function."""
    print("Checking for updates...")

    # Scrape current listings
    soup = scraper.get_soup(SITE_URL)
    current_items = []

    for item in soup.select('.listing-item'):
        current_items.append({
            'id': item.get('data-id'),
            'title': item.select_one('.title').text.strip(),
            'price': item.select_one('.price').text.strip(),
            'url': item.select_one('a')['href']
        })

    # Detect changes
    existing = load_existing()

    detector = ChangeDetector(
        key_fields=['id'],
        compare_fields=['title', 'price']
    )

    changes = detector.detect_changes(current_items, existing)

    # Process new items
    for item in changes['new']:
        aggregator.add_alert(
            title=f"New Listing: {item['title']}",
            message=f"Price: {item['price']}",
            priority=AlertPriority.HIGH,
            metadata={'url': item['url']}
        )

    # Process modified items
    for item in changes['modified']:
        aggregator.add_alert(
            title=f"Updated: {item['title']}",
            message=f"Price changed to: {item['price']}",
            priority=AlertPriority.MEDIUM
        )

    # Send notifications
    pending = aggregator.get_pending_alerts()
    if pending:
        # Build digest
        digest = EmailDigestBuilder(
            f"Site Monitor: {len(pending)} Updates",
            theme=DigestTheme.ALERT
        )
        digest.add_summary({
            "New": len(changes['new']),
            "Modified": len(changes['modified']),
            "Removed": len(changes['removed'])
        })

        for alert in pending:
            router.send(Notification(
                title=alert.title,
                message=alert.message,
                priority=NotificationPriority.HIGH
            ))
            aggregator.mark_sent(alert.id)

    # Save current state
    save_items(current_items)
    print(f"Found {len(changes['new'])} new, {len(changes['modified'])} modified")

# Schedule hourly checks
scheduler = JobScheduler()
scheduler.add_job(Job(
    name="check_site",
    func=check_for_updates,
    schedule=Schedule.hourly(minute=0)
))

if __name__ == "__main__":
    check_for_updates()  # Run once immediately
    scheduler.start(blocking=True)
```

---

## 2. Multi-Source Data Pipeline

Aggregate data from multiple APIs with checkpointing.

### Setup
```bash
python create_project.py --name "Data Pipeline" \
  --patterns connectors-suite,scheduling-suite,export-suite
```

### Code Example

```python
from patterns.connectors import (
    DataSourceRegistry, APISource, FileSource,
    CachedSource, FallbackSource
)
from patterns.scheduling import (
    Pipeline, Task, CheckpointManager
)
from patterns.export import ReportGenerator, ReportFormat

# Setup data sources
registry = DataSourceRegistry()

# Primary API
registry.register("users_api", CachedSource(
    APISource(
        base_url="https://api.example.com",
        auth_token="token-1"
    ),
    ttl_seconds=300
))

# Backup API with fallback
primary = APISource(base_url="https://primary.api.com")
backup = APISource(base_url="https://backup.api.com")
registry.register("orders_api", FallbackSource([primary, backup]))

# Local file source
registry.register("config", FileSource("config/settings.json"))

# Checkpoint for resume capability
checkpoint = CheckpointManager("daily_etl", storage_path="checkpoints/")

def extract_users(context):
    """Extract users from API."""
    result = registry.fetch("users_api", "/users")
    if result.success:
        context['users'] = result.data
        return context
    raise Exception(f"Failed to fetch users: {result.error}")

def extract_orders(context):
    """Extract orders from API."""
    result = registry.fetch("orders_api", "/orders")
    if result.success:
        context['orders'] = result.data
        return context
    raise Exception(f"Failed to fetch orders: {result.error}")

def transform_data(context):
    """Join and transform data."""
    users = {u['id']: u for u in context['users']}

    enriched_orders = []
    for order in context['orders']:
        user = users.get(order['user_id'], {})
        enriched_orders.append({
            'order_id': order['id'],
            'user_name': user.get('name', 'Unknown'),
            'user_email': user.get('email', ''),
            'amount': order['amount'],
            'status': order['status']
        })

    context['enriched_orders'] = enriched_orders
    return context

def load_data(context):
    """Generate reports."""
    report = ReportGenerator("Daily Orders Report")

    # Summary metrics
    orders = context['enriched_orders']
    total_amount = sum(o['amount'] for o in orders)
    completed = len([o for o in orders if o['status'] == 'completed'])

    report.add_metrics("Summary", {
        "Total Orders": len(orders),
        "Completed": completed,
        "Total Amount": f"${total_amount:,.2f}"
    })

    report.add_table("Order Details", orders)

    # Export to multiple formats
    report.export_all("output/daily_report")

    return context

# Build pipeline
pipeline = Pipeline("daily_etl", max_parallel=2)

# Extract in parallel
pipeline.add_task(Task("extract_users", extract_users))
pipeline.add_task(Task("extract_orders", extract_orders))

# Transform after both extracts complete
pipeline.add_task(Task(
    "transform",
    transform_data,
    depends_on=["extract_users", "extract_orders"]
))

# Load after transform
pipeline.add_task(Task("load", load_data, depends_on=["transform"]))

if __name__ == "__main__":
    # Check for existing checkpoint
    state = checkpoint.load()

    if state and state.get('status') == 'partial':
        print("Resuming from checkpoint...")
        result = pipeline.run(resume_from=state.get('last_task'))
    else:
        print("Starting fresh run...")
        result = pipeline.run()

    print(f"Pipeline {result.status.value}")
    print(f"Duration: {result.duration_ms:.0f}ms")

    # Save checkpoint
    if result.status.value == 'completed':
        checkpoint.complete({'status': 'completed'})
    else:
        checkpoint.save({
            'status': 'partial',
            'last_task': list(result.task_results.keys())[-1]
        })
```

---

## 3. Alert Management System

Centralized alert aggregation with multi-channel routing.

### Setup
```bash
python create_project.py --name "Alert Center" \
  --patterns notifications-suite,scheduling-suite
```

### Code Example

```python
from patterns.notifications import (
    AlertAggregator, AlertPriority, AlertBuffer,
    NotificationRouter, Channel, Notification, NotificationPriority,
    SlackHandler, TeamsHandler, ConsoleHandler,
    RoutingRule,
    EmailDigestBuilder, DigestTheme
)
from patterns.scheduling import JobScheduler, Job, Schedule
from datetime import datetime

# Configure aggregator with rate limits
aggregator = AlertAggregator(
    dedup_window_minutes=30,
    max_alerts_per_hour=50,
    quiet_hours=(22, 7),  # 10 PM - 7 AM
    priority_rate_limits={
        AlertPriority.CRITICAL: 0,    # Unlimited
        AlertPriority.HIGH: 20,
        AlertPriority.MEDIUM: 10,
        AlertPriority.LOW: 5
    }
)

# Configure multi-channel router
router = NotificationRouter(retry_count=3)

router.add_channel(Channel.SLACK, SlackHandler(
    webhook_url="https://hooks.slack.com/services/...",
    username="Alert Bot"
))

router.add_channel(Channel.TEAMS, TeamsHandler(
    webhook_url="https://outlook.office.com/webhook/..."
))

router.add_channel(Channel.CONSOLE, ConsoleHandler("[ALERT]"))

# Routing rules
router.add_rule(RoutingRule(
    name="critical_all",
    condition=lambda n: n.priority == NotificationPriority.CRITICAL,
    channels=[Channel.SLACK, Channel.TEAMS],
    immediate=True
))

router.add_rule(RoutingRule(
    name="high_slack",
    condition=lambda n: n.priority == NotificationPriority.HIGH,
    channels=[Channel.SLACK]
))

router.set_default_channels([Channel.CONSOLE])

# Alert sources (simulated)
def check_server_health():
    """Check server metrics and generate alerts."""
    # Simulated metrics
    metrics = {
        'cpu_usage': 85,
        'memory_usage': 92,
        'disk_usage': 78,
        'response_time_ms': 450
    }

    if metrics['cpu_usage'] > 90:
        aggregator.add_alert(
            "High CPU Usage",
            f"CPU at {metrics['cpu_usage']}%",
            AlertPriority.CRITICAL,
            category="infrastructure"
        )
    elif metrics['cpu_usage'] > 80:
        aggregator.add_alert(
            "Elevated CPU Usage",
            f"CPU at {metrics['cpu_usage']}%",
            AlertPriority.MEDIUM,
            category="infrastructure"
        )

    if metrics['memory_usage'] > 90:
        aggregator.add_alert(
            "High Memory Usage",
            f"Memory at {metrics['memory_usage']}%",
            AlertPriority.HIGH,
            category="infrastructure"
        )

    if metrics['response_time_ms'] > 500:
        aggregator.add_alert(
            "Slow Response Time",
            f"Response time: {metrics['response_time_ms']}ms",
            AlertPriority.MEDIUM,
            category="performance"
        )

def process_pending_alerts():
    """Process and route pending alerts."""
    pending = aggregator.get_pending_alerts()

    if not pending:
        return

    print(f"Processing {len(pending)} alerts...")

    for alert in pending:
        notification = Notification(
            title=alert.title,
            message=alert.message,
            priority=NotificationPriority(alert.priority.value),
            category=alert.category,
            metadata=alert.metadata
        )

        results = router.send(notification)

        # Mark as sent if any channel succeeded
        if any(r.success for r in results):
            aggregator.mark_sent(alert.id)
        else:
            print(f"Failed to send alert: {alert.title}")

def send_daily_digest():
    """Send daily digest of all alerts."""
    stats = aggregator.get_stats()
    grouped = aggregator.get_grouped_alerts()

    digest = EmailDigestBuilder(
        f"Daily Alert Digest - {datetime.now().strftime('%Y-%m-%d')}",
        theme=DigestTheme.PROFESSIONAL
    )

    digest.add_summary({
        "Total Alerts": stats['total_alerts'],
        "Critical": stats['by_priority'].get('CRITICAL', 0),
        "High": stats['by_priority'].get('HIGH', 0),
        "Sent Last Hour": stats['sent_last_hour']
    })

    for group_key, group in grouped.items():
        digest.add_alert_section(
            f"Category: {group_key}",
            [{'title': a.title, 'message': a.message, 'severity': a.priority.value}
             for a in group.alerts[:5]]
        )

    html, text = digest.build()
    # Send digest via email...
    print("Daily digest generated")

# Schedule jobs
scheduler = JobScheduler()

scheduler.add_job(Job(
    name="health_check",
    func=check_server_health,
    schedule=Schedule.interval(minutes=5)
))

scheduler.add_job(Job(
    name="process_alerts",
    func=process_pending_alerts,
    schedule=Schedule.interval(minutes=1)
))

scheduler.add_job(Job(
    name="daily_digest",
    func=send_daily_digest,
    schedule=Schedule.daily(hour=8, minute=0)
))

if __name__ == "__main__":
    print("Starting Alert Center...")
    print(f"Router stats: {router.get_stats()}")
    scheduler.start(blocking=True)
```

---

## 4. Business Scoring Dashboard

Score and classify business entities with a web dashboard.

### Setup
```bash
python create_project.py --name "Scorer" \
  --patterns scoring-suite,razors-suite,dashboard,export-suite
```

### Code Example

```python
from patterns.scoring import (
    WeightedScorer, ScoringComponent,
    AlertRulesEngine, AlertRule, AlertSeverity,
    RiskClassifier,
    BenchmarkEngine
)
from patterns.razors import (
    FilterChain, ThresholdFilter,
    TemporalDecay, create_recency_scorer
)
from patterns.export import ReportGenerator, ReportFormat
from flask import Flask, jsonify, render_template

# Scoring configuration
scorer = WeightedScorer([
    ScoringComponent("financial_health", weight=0.30),
    ScoringComponent("operational_efficiency", weight=0.25),
    ScoringComponent("compliance", weight=0.25),
    ScoringComponent("customer_satisfaction", weight=0.20)
])

# Risk classification
risk_classifier = RiskClassifier(
    thresholds={"critical": 30, "high": 50, "medium": 70, "low": 85}
)

# Recency scoring for data freshness
recency_scorer = create_recency_scorer()

# Alert rules
alert_engine = AlertRulesEngine([
    AlertRule(
        name="low_score",
        condition=lambda x: x.get('total_score', 100) < 50,
        severity=AlertSeverity.HIGH,
        message="Score below acceptable threshold"
    ),
    AlertRule(
        name="declining_trend",
        condition=lambda x: x.get('score_change', 0) < -10,
        severity=AlertSeverity.MEDIUM,
        message="Score declining significantly"
    ),
    AlertRule(
        name="compliance_issue",
        condition=lambda x: x.get('compliance', 100) < 60,
        severity=AlertSeverity.CRITICAL,
        message="Compliance score critically low"
    )
])

# Priority filter
priority_filter = ThresholdFilter(
    field='total_score',
    thresholds={'urgent': 40, 'high': 60, 'medium': 75, 'low': 90},
    higher_is_better=True
)

def score_entity(entity_data):
    """Score a single entity."""
    # Calculate component scores
    components = {
        'financial_health': entity_data.get('financial_score', 50),
        'operational_efficiency': entity_data.get('ops_score', 50),
        'compliance': entity_data.get('compliance_score', 50),
        'customer_satisfaction': entity_data.get('csat_score', 50)
    }

    # Get weighted score
    result = scorer.score(components)

    # Classify risk
    risk_level = risk_classifier.classify(result.total_score)

    # Calculate data freshness
    days_old = entity_data.get('days_since_update', 0)
    freshness_score = recency_scorer.score(days_old=days_old)

    # Generate alerts
    alert_data = {
        'total_score': result.total_score,
        'compliance': components['compliance'],
        'score_change': entity_data.get('score_change', 0)
    }
    alerts = alert_engine.evaluate(alert_data)

    # Get priority
    priority = priority_filter.get_priority(result.total_score)

    return {
        'entity_id': entity_data['id'],
        'name': entity_data['name'],
        'total_score': result.total_score,
        'grade': result.grade,
        'risk_level': risk_level.value,
        'priority': priority,
        'freshness_score': freshness_score,
        'components': result.component_scores,
        'alerts': [{'severity': a.severity.value, 'message': a.message} for a in alerts]
    }

def score_all_entities(entities):
    """Score all entities and filter."""
    scored = [score_entity(e) for e in entities]

    # Filter to high priority
    chain = FilterChain()
    chain.where('priority').is_in(['urgent', 'high'])

    high_priority = chain.apply(scored)

    return scored, high_priority

# Flask dashboard
app = Flask(__name__)

# Sample data
SAMPLE_ENTITIES = [
    {'id': 1, 'name': 'Entity A', 'financial_score': 75, 'ops_score': 80,
     'compliance_score': 90, 'csat_score': 85, 'days_since_update': 5},
    {'id': 2, 'name': 'Entity B', 'financial_score': 45, 'ops_score': 50,
     'compliance_score': 55, 'csat_score': 60, 'days_since_update': 30},
    {'id': 3, 'name': 'Entity C', 'financial_score': 85, 'ops_score': 88,
     'compliance_score': 92, 'csat_score': 90, 'days_since_update': 2},
]

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/api/scores')
def get_scores():
    all_scored, high_priority = score_all_entities(SAMPLE_ENTITIES)

    return jsonify({
        'entities': all_scored,
        'high_priority_count': len(high_priority),
        'summary': {
            'total': len(all_scored),
            'avg_score': sum(e['total_score'] for e in all_scored) / len(all_scored),
            'by_risk': {
                'critical': len([e for e in all_scored if e['risk_level'] == 'critical']),
                'high': len([e for e in all_scored if e['risk_level'] == 'high']),
                'medium': len([e for e in all_scored if e['risk_level'] == 'medium']),
                'low': len([e for e in all_scored if e['risk_level'] == 'low'])
            }
        }
    })

@app.route('/api/report')
def generate_report():
    all_scored, _ = score_all_entities(SAMPLE_ENTITIES)

    report = ReportGenerator("Entity Scoring Report")
    report.add_metrics("Summary", {
        "Total Entities": len(all_scored),
        "Average Score": f"{sum(e['total_score'] for e in all_scored) / len(all_scored):.1f}"
    })
    report.add_table("Entity Scores", all_scored)

    paths = report.export_all("output/scoring_report")

    return jsonify({'files': [str(p) for p in paths.values()]})

if __name__ == "__main__":
    app.run(port=5001, debug=True)
```

---

## 5. ML Forecasting Pipeline

Time series forecasting with ensemble models.

### Setup
```bash
python create_project.py --name "Forecaster" \
  --patterns ml-suite,connectors-suite,export-suite
```

### Code Example

```python
from patterns.ml import (
    FeaturePipeline, FeatureCategory,
    TimeSeriesForecaster, ForecastMethod,
    EnsemblePredictor, EnsembleMethod,
    ModelRegistry, ModelStage
)
from patterns.connectors import APISource, DataSourceRegistry
from patterns.export import ReportGenerator, ReportFormat
import pandas as pd
import numpy as np

# Feature engineering
feature_pipeline = FeaturePipeline(categories=[
    FeatureCategory.PRICE,
    FeatureCategory.MOMENTUM,
    FeatureCategory.VOLATILITY,
    FeatureCategory.VOLUME
])

# Model registry
registry = ModelRegistry(storage_path='models/')

# Forecasters
arima_forecaster = TimeSeriesForecaster(method=ForecastMethod.ARIMA)
ets_forecaster = TimeSeriesForecaster(method=ForecastMethod.EXPONENTIAL_SMOOTHING)
ensemble_forecaster = TimeSeriesForecaster(method=ForecastMethod.ENSEMBLE)

def load_historical_data():
    """Load historical time series data."""
    # Simulated data - replace with actual data source
    dates = pd.date_range(start='2023-01-01', periods=365, freq='D')
    np.random.seed(42)

    base = 100
    trend = np.linspace(0, 20, 365)
    seasonal = 10 * np.sin(np.linspace(0, 4*np.pi, 365))
    noise = np.random.randn(365) * 5

    values = base + trend + seasonal + noise

    return pd.DataFrame({
        'date': dates,
        'value': values,
        'volume': np.random.randint(1000, 5000, 365)
    })

def train_models(df):
    """Train and register forecasting models."""
    # Generate features
    features_df = feature_pipeline.generate_features(df)

    # Train individual forecasters
    arima_forecaster.fit(df['value'])
    ets_forecaster.fit(df['value'])
    ensemble_forecaster.fit(df['value'])

    # Register models
    registry.register(
        name='arima_model',
        model=arima_forecaster,
        metrics={'rmse': 5.2, 'mae': 4.1},
        tags=['production', 'v1']
    )

    registry.register(
        name='ets_model',
        model=ets_forecaster,
        metrics={'rmse': 4.8, 'mae': 3.9},
        tags=['production', 'v1']
    )

    registry.register(
        name='ensemble_model',
        model=ensemble_forecaster,
        metrics={'rmse': 4.2, 'mae': 3.5},
        tags=['production', 'v1']
    )

    # Promote best model to production
    registry.transition_stage('ensemble_model', 1, ModelStage.PRODUCTION)

    print("Models trained and registered")

def generate_forecast(horizon=30):
    """Generate forecast using production model."""
    # Load production model
    model = registry.load('ensemble_model', stage=ModelStage.PRODUCTION)

    # Generate forecast with confidence intervals
    result = model.predict_with_intervals(horizon=horizon)

    return {
        'predictions': result.predictions.tolist(),
        'lower_bound': result.lower_bound.tolist(),
        'upper_bound': result.upper_bound.tolist(),
        'confidence_level': result.confidence_level
    }

def generate_forecast_report(forecast, historical_df):
    """Generate forecast report."""
    report = ReportGenerator("Demand Forecast Report")

    # Summary metrics
    report.add_metrics("Forecast Summary", {
        "Horizon": f"{len(forecast['predictions'])} days",
        "Confidence Level": f"{forecast['confidence_level']*100:.0f}%",
        "Predicted Avg": f"{np.mean(forecast['predictions']):.2f}",
        "Predicted Max": f"{max(forecast['predictions']):.2f}"
    })

    # Model comparison
    comparison = registry.compare_versions('ensemble_model')
    report.add_table("Model Versions", [
        {'version': v.version, 'rmse': v.metrics.get('rmse'), 'stage': v.stage.value}
        for v in comparison
    ])

    # Export
    paths = report.export_all("output/forecast_report")
    return paths

if __name__ == "__main__":
    # Load data
    df = load_historical_data()
    print(f"Loaded {len(df)} historical records")

    # Train models
    train_models(df)

    # Generate forecast
    forecast = generate_forecast(horizon=30)
    print(f"Generated {len(forecast['predictions'])}-day forecast")

    # Generate report
    paths = generate_forecast_report(forecast, df)
    print(f"Reports saved: {list(paths.keys())}")
```

---

## 6. Document Generation System

Generate professional documents from data.

### Setup
```bash
python create_project.py --name "DocGen" \
  --patterns export-suite,connectors-suite
```

### Code Example

```python
from patterns.export import (
    ReportGenerator, ReportFormat,
    DocumentBuilder, DocumentStyle, OnePagerBuilder,
    create_proposal_document
)
from patterns.connectors import DataSourceRegistry, FileSource
from pathlib import Path

# Data sources
registry = DataSourceRegistry()
registry.register("projects", FileSource("data/projects.json"))
registry.register("metrics", FileSource("data/metrics.json"))

def generate_executive_summary(project_data, metrics_data):
    """Generate executive summary document."""
    doc = DocumentBuilder(
        title=f"Executive Summary: {project_data['name']}",
        subtitle=f"Prepared for {project_data['client']}",
        style=DocumentStyle.EXECUTIVE
    )

    doc.add_heading("Overview", level=1)
    doc.add_paragraph(project_data['description'])

    doc.add_heading("Key Metrics", level=2)
    doc.add_bullet_list([
        f"ROI: {metrics_data['roi']}%",
        f"Timeline: {metrics_data['timeline']} months",
        f"Budget: ${metrics_data['budget']:,}",
        f"Risk Level: {metrics_data['risk_level']}"
    ])

    doc.add_heading("Recommendations", level=2)
    for rec in project_data['recommendations']:
        doc.add_paragraph(f"• {rec}")

    # Save as Word and Markdown
    doc.save(f"output/{project_data['id']}_summary.docx")
    doc.save(f"output/{project_data['id']}_summary.md")

    return doc

def generate_one_pager(project_data):
    """Generate one-page overview."""
    one_pager = OnePagerBuilder(
        title=project_data['name'],
        subtitle=project_data['tagline']
    )

    one_pager.add_paragraph(project_data['elevator_pitch'])

    one_pager.add_key_value_table({
        "Client": project_data['client'],
        "Industry": project_data['industry'],
        "Timeline": project_data['timeline'],
        "Investment": project_data['investment']
    })

    one_pager.add_bullet_list(project_data['benefits'], title="Key Benefits")

    one_pager.save(f"output/{project_data['id']}_onepager.docx")

    return one_pager

def generate_data_report(data_list, title):
    """Generate multi-format data report."""
    report = ReportGenerator(title)

    # Summary
    report.add_metrics("Overview", {
        "Total Records": len(data_list),
        "Categories": len(set(d.get('category', 'Unknown') for d in data_list))
    })

    # Data table
    report.add_table("Details", data_list)

    # Export to all formats
    paths = report.export_all(f"output/{title.lower().replace(' ', '_')}")

    return paths

def generate_proposal(client_name, project_details):
    """Generate full proposal document."""
    doc = create_proposal_document(
        title=f"Proposal for {client_name}",
        executive_summary=project_details['summary'],
        problem_statement=project_details['problem'],
        proposed_solution=project_details['solution'],
        benefits=project_details['benefits'],
        timeline=[
            {'phase': 'Discovery', 'duration': '2 weeks'},
            {'phase': 'Development', 'duration': '8 weeks'},
            {'phase': 'Testing', 'duration': '2 weeks'},
            {'phase': 'Deployment', 'duration': '1 week'}
        ],
        pricing={
            'base': project_details['base_price'],
            'options': project_details['optional_items']
        }
    )

    doc.save(f"output/proposal_{client_name.lower().replace(' ', '_')}.docx")

    return doc

if __name__ == "__main__":
    # Sample data
    project = {
        'id': 'proj-001',
        'name': 'Digital Transformation Initiative',
        'client': 'Acme Corp',
        'tagline': 'Modernizing operations for the digital age',
        'industry': 'Manufacturing',
        'description': 'A comprehensive digital transformation project...',
        'elevator_pitch': 'Transform your operations with modern technology...',
        'timeline': '6 months',
        'investment': '$500,000',
        'benefits': [
            'Reduce operational costs by 30%',
            'Improve productivity by 50%',
            'Enable real-time decision making'
        ],
        'recommendations': [
            'Implement cloud-first infrastructure',
            'Adopt agile methodology',
            'Invest in employee training'
        ]
    }

    metrics = {
        'roi': 250,
        'timeline': 6,
        'budget': 500000,
        'risk_level': 'Medium'
    }

    # Generate documents
    Path("output").mkdir(exist_ok=True)

    generate_executive_summary(project, metrics)
    print("Executive summary generated")

    generate_one_pager(project)
    print("One-pager generated")

    paths = generate_data_report([project], "Project Overview")
    print(f"Data report generated: {list(paths.keys())}")
```

---

## Next Steps

- Review the [Pattern Reference](PATTERNS.md) for detailed API documentation
- Check the [Quick Start Guide](QUICK-START.md) for setup instructions
- Explore the `.claude/framework-context.md` for the complete framework guide
