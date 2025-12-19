# Using the Framework with Claude Code

## Quick Reference for Prompting Claude Code

This guide shows you exactly how to use this framework with Claude Code to rapidly build new business solutions.

---

## Setup (One Time Only)

1. Open this framework directory in Claude Code
2. Claude will automatically read `.claude/framework-context.md`
3. Claude now understands all available patterns and how to use them

That's it! The framework is ready to use.

---

## Basic Usage Pattern

### 1. Describe Your Business Problem

Simply tell Claude Code what you want to build:

```
"I need a tool that tracks [WHAT] from [WHERE] and alerts me when [CONDITION]."
```

### 2. Reference the Framework

Tell Claude to use the framework:

```
"Using the universal-business-solution-framework, build this for me."
```

### 3. Specify Patterns (Optional)

If you know which patterns you need:

```
"Use patterns: scraping + change-detection + email + dashboard"
```

---

## Example Prompts (Copy & Paste)

### Example 1: Web Monitoring System

```
Using the universal-business-solution-framework, build a tool that:

- Scrapes job postings from [URL]
- Detects when new jobs matching "Python Developer" are posted
- Sends me email alerts immediately
- Shows a dashboard with all current postings

Use patterns: scraping + change-detection + email + dashboard + scheduler

Run the scraper every 30 minutes.
```

### Example 2: Data Analysis Tool

```
Using the universal-business-solution-framework, build a tool that:

- Processes a CSV file of sales data
- Analyzes trends by region and product
- Generates an executive summary report
- Creates a dashboard showing key metrics

Use patterns: analysis + documentation + dashboard
```

### Example 3: Price Monitoring

```
Using the universal-business-solution-framework, build a tool that:

- Scrapes product prices from [competitor website]
- Tracks when prices change
- Alerts me when price drops below $X
- Charts price history over time

Use patterns: scraping + change-detection + email + dashboard + scheduler
```

### Example 4: Compliance Monitoring

```
Using the universal-business-solution-framework, build a tool that:

- Checks government website for new regulations
- Detects when new rules are published
- Sends Slack notifications to compliance team
- Maintains history of all changes

Use patterns: scraping + change-detection + email + scheduler
```

### Example 5: Market Research Automation

```
Using the universal-business-solution-framework, build a tool that:

- Scrapes competitor product listings daily
- Analyzes pricing and features
- Generates weekly summary report
- Sends PDF report via email

Use patterns: scraping + analysis + documentation + scheduler + email
```

---

## Two-Step Approach (Recommended)

For more control, use a two-step approach:

### Step 1: Generate Project Structure

```bash
python create_project.py --name "NYC Building Permits Tracker" --patterns scraping,change-detection,email,dashboard
```

### Step 2: Customize in Claude Code

Open the generated project and prompt Claude:

```
This project was generated from the universal-business-solution-framework.

Customize it to:
- Scrape NYC building permits from [URL]
- Extract fields: address, permit_type, date_filed, status
- Alert when permits are filed in zip code 10001
- Dashboard should show permits by type and neighborhood

The current scraper is in src/scraping_patterns.py - modify it for this use case.
```

---

## Advanced Usage

### Combining Multiple Data Sources

```
Using the universal-business-solution-framework:

Build a tool that combines data from:
1. API: Weather data from OpenWeather API
2. Scraping: Event listings from [event website]
3. File: Historical attendance CSV

Analyze correlation between weather and attendance.
Generate weekly report with recommendations.

Use patterns: scraping + analysis + documentation + scheduler
```

### Custom Pattern Assembly

```
Using the universal-business-solution-framework:

I need a custom solution with these capabilities:
- Data source: REST API (not scraping)
- Processing: Real-time stream processing (not batch)
- Alerts: Slack + Email
- Storage: PostgreSQL database

Base it on the scheduler and email patterns, but customize the data source.
```

---

## Pattern Selection Guide

Use this to choose the right patterns for your project:

| If You Need... | Use Pattern(s) |
|---|---|
| Extract data from website | `scraping` |
| Track changes over time | `change-detection` |
| Send notifications | `email` |
| Visual monitoring | `dashboard` |
| Run automatically | `scheduler` |
| Process/analyze data | `analysis` |
| Create reports | `documentation` |

### Common Combinations

| Project Type | Patterns |
|---|---|
| Monitoring System | `scraping` + `change-detection` + `email` + `dashboard` + `scheduler` |
| Analysis Tool | `analysis` + `documentation` + `dashboard` |
| Data Pipeline | `scraping` + `scheduler` + `email` |
| Alert System | `change-detection` + `email` + `scheduler` |
| Reporting Tool | `analysis` + `documentation` + `email` + `scheduler` |

---

## Template Prompts

### For Web Scraping Projects

```
Using the universal-business-solution-framework:

Scrape [URL] for [data type].
Key fields: [field1], [field2], [field3]
Update frequency: [interval]
Alert when: [condition]

Use patterns: scraping + change-detection + email + scheduler
```

### For Data Analysis Projects

```
Using the universal-business-solution-framework:

Analyze [data source] for [insights].
Key metrics: [metric1], [metric2]
Output: [report type]
Frequency: [schedule]

Use patterns: analysis + documentation + email + scheduler
```

### For Monitoring Projects

```
Using the universal-business-solution-framework:

Monitor [source] for [events].
Alert when: [conditions]
Dashboard showing: [visualizations]
Check every: [interval]

Use patterns: scraping + change-detection + email + dashboard + scheduler
```

---

## Best Practices for Prompting

### ✅ DO

- Be specific about your data source (URL, API, file path)
- Describe the business problem, not just the technical solution
- Specify key data fields you care about
- Mention alert conditions clearly
- State update frequency/schedule
- Reference pattern names from the framework

### ❌ DON'T

- Be vague ("build me a scraper")
- Skip the business context
- Forget to specify which patterns to use
- Ask for features not in the patterns
- Assume Claude knows your specific website structure

### Good Example

```
Using the universal-business-solution-framework:

Build a procurement opportunity tracker that:
- Scrapes https://procurement.example.gov/opportunities
- Extracts: title, deadline, value, requirements
- Alerts me when opportunities over $50K are posted
- Shows dashboard with opportunities by department
- Runs every 2 hours

Use patterns: scraping + change-detection + email + dashboard + scheduler

The table has id="opportunities-table" and headers in row 1.
```

### Bad Example

```
Build a scraper for a government website
```
*(Too vague - no URL, no data fields, no patterns specified)*

---

## Iterative Development

Start simple, then enhance:

### Iteration 1: Basic Scraping
```
Using the universal-business-solution-framework:
Build a basic scraper for [URL] that extracts [data] and saves to CSV.
Use pattern: scraping
```

### Iteration 2: Add Change Detection
```
Now add change detection to track new items.
Use the change-detection pattern.
```

### Iteration 3: Add Alerts
```
Add email alerts when new items are detected.
Use the email pattern with Gmail SMTP.
```

### Iteration 4: Add Dashboard
```
Create a web dashboard to visualize the data.
Use the dashboard pattern with Chart.js.
```

### Iteration 5: Automate
```
Schedule this to run every hour.
Use the scheduler pattern.
```

---

## Troubleshooting Prompts

### If Claude doesn't find the framework:

```
Please read the file at:
C:\Users\dada_\OneDrive\Documents\universal-business-solution-framework\.claude\framework-context.md

Then use that framework to build my solution.
```

### If you need to see available patterns:

```
List all available patterns from the universal-business-solution-framework.
```

### If you want to see an example:

```
Show me how the NYPD Procurement Tracker uses the framework patterns.
Reference the example at:
C:\Users\dada_\OneDrive\Documents\nypd-procurement-tracker\
```

---

## Quick Wins (30 Minutes or Less)

### 1. Job Posting Tracker
```
Using universal-business-solution-framework:
Track job postings from [careers page], alert on "Senior" + "Python" roles.
Patterns: scraping + change-detection + email
```

### 2. Price Monitor
```
Using universal-business-solution-framework:
Monitor product price on [e-commerce site], alert on drops > 10%.
Patterns: scraping + change-detection + email + scheduler
```

### 3. News Aggregator
```
Using universal-business-solution-framework:
Scrape news from [site], filter by keywords, email daily digest.
Patterns: scraping + email + scheduler
```

---

## Next Steps After Claude Builds Your Project

1. **Review the generated code** - Understand what was created
2. **Install dependencies** - `pip install -r requirements.txt`
3. **Configure environment** - Copy `.env.example` to `.env` and fill in
4. **Test manually** - Run the scraper/analyzer once
5. **Verify alerts** - Send a test email
6. **View dashboard** - Open http://localhost:5001
7. **Schedule automated runs** - Start the scheduler
8. **Monitor for 24 hours** - Ensure it works as expected
9. **Iterate and improve** - Ask Claude to add features

---

**Remember**: The framework does the heavy lifting. You just describe what you want to build!
