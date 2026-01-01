"""
Universal Business Solution Framework - Project Generator

Quick project generator using framework patterns

Usage:
    python create_project.py --name "My Project" --patterns scraping,dashboard,email
    python create_project.py --name "NYC Permits Tracker" --patterns scraping,change-detection,email,scheduler
    python create_project.py --interactive  # Interactive mode
"""

import argparse
import os
from pathlib import Path
import shutil


# Available patterns and their descriptions
AVAILABLE_PATTERNS = {
    # === Core Patterns ===
    'scraping': {
        'name': 'Web Scraping',
        'description': 'Extract data from websites using BeautifulSoup',
        'files': ['scraping_patterns.py'],
        'dependencies': ['requests', 'beautifulsoup4', 'lxml', 'pandas'],
        'category': 'core'
    },
    'change-detection': {
        'name': 'Change Detection',
        'description': 'Track changes over time with baseline comparison',
        'files': ['change_detection_patterns.py'],
        'dependencies': [],
        'category': 'core'
    },
    'email': {
        'name': 'Email Alerts',
        'description': 'Send HTML email notifications',
        'files': ['email_patterns.py'],
        'dependencies': [],
        'category': 'core'
    },
    'scheduler': {
        'name': 'Task Scheduling',
        'description': 'Run tasks automatically at intervals',
        'files': ['scheduler_patterns.py'],
        'dependencies': ['schedule'],
        'category': 'core'
    },
    'dashboard': {
        'name': 'Web Dashboard',
        'description': 'Flask web interface with real-time updates',
        'files': [],  # Handled separately
        'dependencies': ['flask', 'flask-cors'],
        'category': 'core'
    },
    'analysis': {
        'name': 'Data Analysis',
        'description': 'Process and analyze data with pandas',
        'files': [],
        'dependencies': ['pandas', 'numpy'],
        'category': 'core'
    },
    'documentation': {
        'name': 'Documentation Generation',
        'description': 'Generate Word docs and markdown',
        'files': [],
        'dependencies': ['python-docx'],
        'category': 'core'
    },

    # === Scoring & Risk Patterns ===
    'weighted-scoring': {
        'name': 'Weighted Scoring Engine',
        'description': 'Multi-component weighted scoring with grades and risk levels',
        'files': ['scoring/weighted_scoring.py'],
        'dependencies': [],
        'category': 'scoring'
    },
    'alert-rules': {
        'name': 'Alert Rules Engine',
        'description': 'Declarative rule-based alert generation with severity levels',
        'files': ['scoring/alert_rules_engine.py'],
        'dependencies': [],
        'category': 'scoring'
    },
    'risk-classification': {
        'name': 'Risk Classification',
        'description': 'Convert scores to risk levels with configurable thresholds',
        'files': ['scoring/risk_classification.py'],
        'dependencies': [],
        'category': 'scoring'
    },
    'benchmarking': {
        'name': 'KPI Benchmarking',
        'description': 'Compare KPIs against benchmarks with gap analysis',
        'files': ['scoring/benchmark_engine.py'],
        'dependencies': [],
        'category': 'scoring'
    },
    'scoring-suite': {
        'name': 'Complete Scoring Suite',
        'description': 'All scoring patterns: weighted scoring, alerts, risk, benchmarks',
        'files': [
            'scoring/__init__.py',
            'scoring/weighted_scoring.py',
            'scoring/alert_rules_engine.py',
            'scoring/risk_classification.py',
            'scoring/benchmark_engine.py'
        ],
        'dependencies': [],
        'category': 'scoring'
    },

    # === ML/AI Patterns ===
    'feature-pipeline': {
        'name': 'Feature Engineering Pipeline',
        'description': 'Technical indicators, statistical features, custom transformations',
        'files': ['ml/feature_pipeline.py'],
        'dependencies': ['pandas', 'numpy'],
        'category': 'ml'
    },
    'time-series': {
        'name': 'Time Series Forecasting',
        'description': 'ARIMA, exponential smoothing, ensemble forecasting',
        'files': ['ml/time_series_models.py'],
        'dependencies': ['pandas', 'numpy'],
        'category': 'ml'
    },
    'ensemble': {
        'name': 'Ensemble Predictor',
        'description': 'Multi-model prediction with weighted voting and stacking',
        'files': ['ml/ensemble_predictor.py'],
        'dependencies': ['pandas', 'numpy'],
        'category': 'ml'
    },
    'model-registry': {
        'name': 'Model Registry',
        'description': 'Model versioning, storage, and lifecycle management',
        'files': ['ml/model_registry.py'],
        'dependencies': [],
        'category': 'ml'
    },
    'ml-suite': {
        'name': 'Complete ML Suite',
        'description': 'All ML patterns: features, forecasting, ensemble, registry',
        'files': [
            'ml/__init__.py',
            'ml/feature_pipeline.py',
            'ml/time_series_models.py',
            'ml/ensemble_predictor.py',
            'ml/model_registry.py'
        ],
        'dependencies': ['pandas', 'numpy'],
        'category': 'ml'
    },

    # === Razors (Analysis Patterns) ===
    'temporal-decay': {
        'name': 'Temporal Decay Scoring',
        'description': 'Time-based scoring with recency brackets and expiration tracking',
        'files': ['razors/temporal_decay.py'],
        'dependencies': [],
        'category': 'razors'
    },
    'filter-chains': {
        'name': 'Filter Chains',
        'description': 'Composable filter logic with AND/OR, negative filters, thresholds',
        'files': ['razors/filter_chains.py'],
        'dependencies': [],
        'category': 'razors'
    },
    'deduplication': {
        'name': 'Deduplication & Change Detection',
        'description': 'Multi-key matching, change tracking, fingerprinting',
        'files': ['razors/deduplication.py'],
        'dependencies': [],
        'category': 'razors'
    },
    'razors-suite': {
        'name': 'Complete Razors Suite',
        'description': 'All analysis patterns: temporal decay, filters, deduplication',
        'files': [
            'razors/__init__.py',
            'razors/temporal_decay.py',
            'razors/filter_chains.py',
            'razors/deduplication.py'
        ],
        'dependencies': [],
        'category': 'razors'
    },

    # === Export & Reporting ===
    'report-generator': {
        'name': 'Report Generator',
        'description': 'Multi-format reports: CSV, JSON, Excel, HTML, Markdown',
        'files': ['export/report_generator.py'],
        'dependencies': ['pandas', 'openpyxl'],
        'category': 'export'
    },
    'document-builder': {
        'name': 'Document Builder',
        'description': 'Professional Word/HTML documents with styling',
        'files': ['export/document_builder.py'],
        'dependencies': ['python-docx'],
        'category': 'export'
    },
    'export-suite': {
        'name': 'Complete Export Suite',
        'description': 'All export patterns: reports, documents',
        'files': [
            'export/__init__.py',
            'export/report_generator.py',
            'export/document_builder.py'
        ],
        'dependencies': ['pandas', 'openpyxl', 'python-docx'],
        'category': 'export'
    },

    # === Notifications ===
    'email-digest': {
        'name': 'Email Digest Builder',
        'description': 'Professional HTML email templates with themes and sections',
        'files': ['notifications/email_digest.py'],
        'dependencies': [],
        'category': 'notifications'
    },
    'alert-aggregator': {
        'name': 'Alert Aggregator',
        'description': 'Deduplicate, rate-limit, and group alerts before sending',
        'files': ['notifications/alert_aggregator.py'],
        'dependencies': [],
        'category': 'notifications'
    },
    'notification-router': {
        'name': 'Notification Router',
        'description': 'Route notifications to Email, Slack, Teams, Webhooks',
        'files': ['notifications/notification_router.py'],
        'dependencies': [],
        'category': 'notifications'
    },
    'notifications-suite': {
        'name': 'Complete Notifications Suite',
        'description': 'All notification patterns: digests, aggregation, routing',
        'files': [
            'notifications/__init__.py',
            'notifications/email_digest.py',
            'notifications/alert_aggregator.py',
            'notifications/notification_router.py'
        ],
        'dependencies': [],
        'category': 'notifications'
    },

    # === Data Connectors ===
    'api-client': {
        'name': 'API Client',
        'description': 'Robust API client with rate limiting, retry, caching',
        'files': ['connectors/api_client.py'],
        'dependencies': [],
        'category': 'connectors'
    },
    'scraper-base': {
        'name': 'Scraper Base',
        'description': 'Web scraping with sessions, anti-detection, proxy support',
        'files': ['connectors/scraper_base.py'],
        'dependencies': ['beautifulsoup4'],
        'category': 'connectors'
    },
    'data-source-registry': {
        'name': 'Data Source Registry',
        'description': 'Unified interface for APIs, databases, files, scrapers',
        'files': ['connectors/data_source_registry.py'],
        'dependencies': [],
        'category': 'connectors'
    },
    'connectors-suite': {
        'name': 'Complete Connectors Suite',
        'description': 'All connector patterns: API client, scraper, registry',
        'files': [
            'connectors/__init__.py',
            'connectors/api_client.py',
            'connectors/scraper_base.py',
            'connectors/data_source_registry.py'
        ],
        'dependencies': ['beautifulsoup4'],
        'category': 'connectors'
    },

    # === Scheduling & Orchestration ===
    'job-scheduler': {
        'name': 'Job Scheduler',
        'description': 'Cron-like scheduling with dependencies and retry logic',
        'files': ['scheduling/job_scheduler.py'],
        'dependencies': [],
        'category': 'scheduling'
    },
    'pipeline-orchestrator': {
        'name': 'Pipeline Orchestrator',
        'description': 'DAG-based task pipelines with parallel execution',
        'files': ['scheduling/pipeline_orchestrator.py'],
        'dependencies': [],
        'category': 'scheduling'
    },
    'checkpoint-manager': {
        'name': 'Checkpoint Manager',
        'description': 'Resume from failure with checkpoint/restore',
        'files': ['scheduling/checkpoint_manager.py'],
        'dependencies': [],
        'category': 'scheduling'
    },
    'scheduling-suite': {
        'name': 'Complete Scheduling Suite',
        'description': 'All scheduling patterns: jobs, pipelines, checkpoints',
        'files': [
            'scheduling/__init__.py',
            'scheduling/job_scheduler.py',
            'scheduling/pipeline_orchestrator.py',
            'scheduling/checkpoint_manager.py'
        ],
        'dependencies': [],
        'category': 'scheduling'
    },
}


def create_project_structure(project_dir):
    """Create base project directory structure"""
    directories = [
        'src',
        'data',
        'output',
        'docs',
        'tests'
    ]

    for dir_name in directories:
        (project_dir / dir_name).mkdir(parents=True, exist_ok=True)

    print(f"[OK] Created project structure in {project_dir}")


def copy_pattern_files(patterns, project_dir):
    """Copy pattern files to project src directory"""
    framework_dir = Path(__file__).parent
    patterns_dir = framework_dir / 'patterns'

    for pattern in patterns:
        if pattern in AVAILABLE_PATTERNS:
            for file in AVAILABLE_PATTERNS[pattern]['files']:
                src_file = patterns_dir / file
                if src_file.exists():
                    # Handle subdirectories (e.g., scoring/weighted_scoring.py)
                    dest_file = project_dir / 'src' / file
                    dest_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(src_file, dest_file)
                    print(f"[OK] Copied {file}")


def create_dashboard(project_dir, project_name):
    """Create Flask dashboard structure"""
    web_dir = project_dir / 'web'
    web_dir.mkdir(exist_ok=True)

    # Create subdirectories
    (web_dir / 'templates').mkdir(exist_ok=True)
    (web_dir / 'static' / 'css').mkdir(parents=True, exist_ok=True)
    (web_dir / 'static' / 'js').mkdir(parents=True, exist_ok=True)

    # Create basic Flask app
    app_content = f'''"""
{project_name} - Web Dashboard
Real-time monitoring dashboard
"""
from flask import Flask, render_template, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')


@app.route('/api/data')
def get_data():
    """API endpoint for dashboard data"""
    return jsonify({{
        'success': True,
        'data': []
    }})


if __name__ == '__main__':
    print("=" * 70)
    print("{project_name} - Web Dashboard")
    print("=" * 70)
    print("Starting server...")
    print("Dashboard will be available at: http://localhost:5001")
    print("Press Ctrl+C to stop")
    print("=" * 70)

    app.run(debug=True, host='0.0.0.0', port=5001)
'''

    with open(web_dir / 'app.py', 'w') as f:
        f.write(app_content)

    print("[OK] Created Flask dashboard structure")


def generate_requirements(patterns, project_dir):
    """Generate requirements.txt with pattern dependencies"""
    dependencies = set()

    for pattern in patterns:
        if pattern in AVAILABLE_PATTERNS:
            dependencies.update(AVAILABLE_PATTERNS[pattern]['dependencies'])

    # Add base dependencies
    base_deps = ['python-dotenv>=1.0.0']
    dependencies.update(base_deps)

    requirements_content = '\n'.join(sorted(dependencies))

    with open(project_dir / 'requirements.txt', 'w') as f:
        f.write(requirements_content)

    print(f"[OK] Generated requirements.txt with {len(dependencies)} dependencies")


def generate_readme(project_name, patterns, project_dir):
    """Generate README.md"""
    patterns_list = '\n'.join([f"- {AVAILABLE_PATTERNS[p]['name']}" for p in patterns if p in AVAILABLE_PATTERNS])

    readme_content = f"""# {project_name}

## Overview

{project_name} - Built using the Universal Business Solution Framework

## Patterns Used

{patterns_list}

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configuration

Copy `.env.example` to `.env` and fill in your configuration:

```bash
copy .env.example .env
```

### 3. Run the Application

{"```bash\npython web/app.py\n```" if 'dashboard' in patterns else "```bash\npython src/main.py\n```"}

## Project Structure

```
{project_dir.name}/
├── src/                # Source code
├── web/                # Web dashboard (if applicable)
├── data/               # Data storage
├── output/             # Exports and reports
├── docs/               # Documentation
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Features

- TODO: List your features here

## Documentation

- TODO: Add links to documentation

## License

MIT License
"""

    with open(project_dir / 'README.md', 'w') as f:
        f.write(readme_content)

    print("[OK] Generated README.md")


def generate_env_example(patterns, project_dir):
    """Generate .env.example"""
    env_content = """# Environment Configuration

# Email Configuration (if using email pattern)
ALERT_EMAIL=your-email@example.com
EMAIL_USERNAME=your-email@gmail.com
EMAIL_PASSWORD=your-app-password
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587

# API Keys (if needed)
# API_KEY=your-api-key-here

# Database (if needed)
# DATABASE_URL=sqlite:///data/database.db
"""

    with open(project_dir / '.env.example', 'w') as f:
        f.write(env_content)

    print("[OK] Generated .env.example")


def generate_main_script(patterns, project_dir, project_name):
    """Generate main.py starter script"""
    imports = []
    code = []

    # Add imports based on patterns
    if 'scraping' in patterns:
        imports.append("from scraping_patterns import UniversalScraper")
        code.append("""
    # Initialize scraper
    scraper = UniversalScraper(url="https://example.com/data")
    data = scraper.scrape()
    scraper.save_to_csv()
    scraper.print_summary()
""")

    if 'change-detection' in patterns:
        imports.append("from change_detection_patterns import ChangeDetector")
        code.append("""
    # Detect changes
    detector = ChangeDetector(data_dir='data')
    changes = detector.detect_changes(data, key_fields=['id'])
    print(detector.format_change_report(changes))
""")

    if 'email' in patterns:
        imports.append("from email_patterns import EmailAlerter")
        imports.append("import os")
        code.append("""
    # Send email alerts
    if changes['new_count'] > 0:
        alerter = EmailAlerter()
        alerter.send_new_items_alert(
            items=changes['new_items'],
            to_email=os.getenv('ALERT_EMAIL'),
            title="New Items Detected"
        )
""")

    main_content = f'''"""
{project_name} - Main Script
"""
{chr(10).join(imports)}


def main():
    """Main execution"""
    print("=" * 70)
    print("{project_name}")
    print("=" * 70)

{''.join(code) if code else '    # TODO: Add your main logic here\n    pass'}

    print("=" * 70)
    print("[OK] Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
'''

    with open(project_dir / 'src' / 'main.py', 'w') as f:
        f.write(main_content)

    print("[OK] Generated src/main.py")


def create_project(name, patterns, output_dir=None):
    """Generate complete project structure"""
    # Create project directory
    project_name = name.lower().replace(' ', '-')
    if output_dir:
        project_dir = Path(output_dir) / project_name
    else:
        project_dir = Path.cwd().parent / project_name

    # Create structure
    create_project_structure(project_dir)

    # Copy pattern files
    copy_pattern_files(patterns, project_dir)

    # Generate files
    generate_requirements(patterns, project_dir)
    generate_readme(name, patterns, project_dir)
    generate_env_example(patterns, project_dir)
    generate_main_script(patterns, project_dir, name)

    # Create dashboard if requested
    if 'dashboard' in patterns:
        create_dashboard(project_dir, name)

    # Final summary
    print("\n" + "=" * 70)
    print(f"[SUCCESS] Project '{name}' created successfully!")
    print("=" * 70)
    print(f"Location: {project_dir}")
    print(f"Patterns: {', '.join(patterns)}")
    print("\nNext steps:")
    print(f"1. cd {project_dir}")
    print("2. pip install -r requirements.txt")
    print("3. copy .env.example .env (and configure)")
    if 'dashboard' in patterns:
        print("4. python web/app.py")
    else:
        print("4. python src/main.py")
    print("=" * 70)

    return project_dir


def interactive_mode():
    """Interactive project creation"""
    print("=" * 70)
    print("UNIVERSAL BUSINESS SOLUTION FRAMEWORK - Project Generator")
    print("=" * 70)
    print()

    # Get project name
    name = input("Project name: ").strip()
    if not name:
        print("[ERROR] Project name is required")
        return

    # Show available patterns by category
    print("\nAvailable patterns:")

    # Group patterns by category
    categories = {}
    for key, info in AVAILABLE_PATTERNS.items():
        cat = info.get('category', 'other')
        if cat not in categories:
            categories[cat] = []
        categories[cat].append((key, info))

    for cat, pattern_list in categories.items():
        print(f"\n  [{cat.upper()}]")
        for key, info in pattern_list:
            print(f"    {key:22} - {info['description']}")

    print()
    patterns_input = input("Patterns (comma-separated, e.g., scraping,email,scoring-suite): ").strip()
    patterns = [p.strip() for p in patterns_input.split(',') if p.strip()]

    # Validate patterns
    invalid = [p for p in patterns if p not in AVAILABLE_PATTERNS]
    if invalid:
        print(f"[ERROR] Invalid patterns: {', '.join(invalid)}")
        return

    print()
    create_project(name, patterns)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Generate a new project using Universal Business Solution Framework'
    )
    parser.add_argument('--name', type=str, help='Project name')
    parser.add_argument('--patterns', type=str, help='Comma-separated list of patterns (e.g., scraping,email,dashboard)')
    parser.add_argument('--output-dir', type=str, help='Output directory (default: parent of current directory)')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    parser.add_argument('--list-patterns', action='store_true', help='List available patterns')

    args = parser.parse_args()

    if args.list_patterns:
        print("\nAvailable Patterns:")
        print("=" * 70)
        for key, info in AVAILABLE_PATTERNS.items():
            print(f"{key:20} - {info['description']}")
        print("=" * 70)
        return

    if args.interactive:
        interactive_mode()
        return

    if not args.name or not args.patterns:
        parser.print_help()
        print("\nTip: Use --interactive for guided project creation")
        return

    patterns = [p.strip() for p in args.patterns.split(',')]
    create_project(args.name, patterns, args.output_dir)


if __name__ == "__main__":
    main()
