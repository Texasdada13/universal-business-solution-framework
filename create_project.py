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
    'scraping': {
        'name': 'Web Scraping',
        'description': 'Extract data from websites using BeautifulSoup',
        'files': ['scraping_patterns.py'],
        'dependencies': ['requests', 'beautifulsoup4', 'lxml', 'pandas']
    },
    'change-detection': {
        'name': 'Change Detection',
        'description': 'Track changes over time with baseline comparison',
        'files': ['change_detection_patterns.py'],
        'dependencies': []
    },
    'email': {
        'name': 'Email Alerts',
        'description': 'Send HTML email notifications',
        'files': ['email_patterns.py'],
        'dependencies': []
    },
    'scheduler': {
        'name': 'Task Scheduling',
        'description': 'Run tasks automatically at intervals',
        'files': ['scheduler_patterns.py'],
        'dependencies': ['schedule']
    },
    'dashboard': {
        'name': 'Web Dashboard',
        'description': 'Flask web interface with real-time updates',
        'files': [],  # Handled separately
        'dependencies': ['flask', 'flask-cors']
    },
    'analysis': {
        'name': 'Data Analysis',
        'description': 'Process and analyze data with pandas',
        'files': [],
        'dependencies': ['pandas', 'numpy']
    },
    'documentation': {
        'name': 'Documentation Generation',
        'description': 'Generate Word docs and markdown',
        'files': [],
        'dependencies': ['python-docx']
    }
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
                    dest_file = project_dir / 'src' / file
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

    # Show available patterns
    print("\nAvailable patterns:")
    for key, info in AVAILABLE_PATTERNS.items():
        print(f"  {key:20} - {info['description']}")

    print()
    patterns_input = input("Patterns (comma-separated, e.g., scraping,email,dashboard): ").strip()
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
