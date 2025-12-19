# Universal Business Solution Framework - Quick Reference

## One-Minute Overview

**What**: Reusable patterns to build business tools fast
**Where**: `C:\Users\dada_\OneDrive\Documents\universal-business-solution-framework\`
**How**: Combine patterns like building blocks

---

## Create New Project (30 Seconds)

```bash
cd C:\Users\dada_\OneDrive\Documents\universal-business-solution-framework

# Interactive
python create_project.py --interactive

# Direct
python create_project.py --name "My Tool" --patterns scraping,email,dashboard
```

---

## 7 Core Patterns

| Pattern | What It Does | When To Use |
|---------|--------------|-------------|
| `scraping` | Extract data from websites | Need data from web pages |
| `change-detection` | Track what's new/changed | Monitor for updates |
| `email` | Send alerts | Notify stakeholders |
| `dashboard` | Web interface | Visual monitoring |
| `scheduler` | Run automatically | Periodic execution |
| `analysis` | Process data | Find insights |
| `documentation` | Generate reports | Executive summaries |

---

## Common Combinations

```bash
# Web Monitoring
--patterns scraping,change-detection,email,dashboard,scheduler

# Business Analysis
--patterns analysis,documentation,dashboard

# Data Pipeline
--patterns scraping,scheduler,email

# Alert System
--patterns change-detection,email,scheduler
```

---

## Claude Code Prompt Template

```
Using the universal-business-solution-framework:

Build a tool that:
- [What it does]
- [Where it gets data]
- [When to alert]
- [How to display]

Use patterns: [pattern1] + [pattern2] + [pattern3]
```

---

## Example: 5-Minute Project

```bash
# 1. Generate project (30 seconds)
python create_project.py --name "Price Tracker" --patterns scraping,change-detection,email

# 2. Install dependencies (1 minute)
cd price-tracker
pip install -r requirements.txt

# 3. Configure (1 minute)
copy .env.example .env
# Edit .env with your email

# 4. Customize in Claude Code (2 minutes)
# Open project, prompt: "Customize this to track prices from [URL]"

# 5. Run
python src/main.py
```

---

## Files You'll Use Most

| File | Purpose |
|------|---------|
| `.claude/framework-context.md` | Teaches Claude about framework |
| `create_project.py` | Generate new projects |
| `patterns/*.py` | Copy/paste reusable code |
| `CLAUDE_CODE_GUIDE.md` | How to prompt Claude |

---

## Example Projects

- **NYPD Procurement Tracker** - `nypd-procurement-tracker/`
- **Application Rationalization** - `application-rationalization-tool/`
- **Capital Projects Planner** - `application-rationalization-tool/capital_projects/`

---

## Common Commands

```bash
# List patterns
python create_project.py --list-patterns

# Create project
python create_project.py --name "X" --patterns scraping,email

# Install dependencies
pip install -r requirements.txt

# Run dashboard
python web/app.py

# Run scheduler
python src/scheduler.py --interval 60
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Import errors | `pip install -r requirements.txt` |
| Port conflict | Change port in `web/app.py` |
| Email not sending | Check `.env` SMTP settings |
| Pattern not found | `python create_project.py --list-patterns` |

---

## Time Estimates

| Project Type | Time to Build |
|--------------|---------------|
| Basic scraper | 30 min |
| Monitoring system | 2-4 hours |
| Analysis tool | 4-6 hours |
| Full dashboard | 3-5 hours |

---

## Pattern Selection Cheat Sheet

**Need to get data from web?** → `scraping`
**Need to track changes?** → `change-detection`
**Need notifications?** → `email`
**Need visual interface?** → `dashboard`
**Need automation?** → `scheduler`
**Need to analyze data?** → `analysis`
**Need reports?** → `documentation`

---

## Next Steps

1. Read `CLAUDE_CODE_GUIDE.md` for detailed prompts
2. Generate a test project: `python create_project.py --interactive`
3. Review example projects for reference
4. Start building!

---

**Framework Location**: `C:\Users\dada_\OneDrive\Documents\universal-business-solution-framework\`
**Version**: 1.0
**License**: MIT
