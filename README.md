# AI Data Analysis Workspace

An AI-powered data analysis platform for CSV datasets built with Streamlit and OpenAI.

This project enables users to upload datasets, explore data quality, ask analytical questions in natural language, automatically generate visualisations, receive AI-generated chart explanations, and export analytical reports.

https://ai-data-analysis-workspace-j8qcctkchziuqghezkaxjm.streamlit.app/

---

# Features

## Dataset Upload

* Upload CSV datasets directly through the web interface
* Automatic dataset preview and schema detection
* Support for numeric and categorical data exploration

## Data Quality Analysis

* Missing value detection
* Duplicate row detection
* Constant column identification
* Column profiling and data type inspection

## AI Assistant

* Ask questions about uploaded datasets using natural language
* Generate Python or R analysis guidance
* Request business insights and analytical explanations
* Multi-turn conversation support

## AI Chart Generator

Generate charts using natural language prompts.

Supported visualisations:

* Line charts
* Bar charts
* Grouped bar charts
* Stacked bar charts
* Scatter plots
* Pie charts
* Histograms
* Boxplots
* Area charts
* Heatmaps

Example prompts:

```text
Show migration trends over time.
```

```text
Show the distribution of estimate by visa using a boxplot, excluding totals and only include estimate values under 2000.
```

```text
Show which visa categories contribute the most, split by citizenship, excluding totals.
```

## AI Chart Explanation

* Automatically explains generated visualisations
* Highlights trends, categories, comparisons, and patterns
* Provides practical analytical interpretation

## Suggested Analysis Questions

* Automatically recommends useful analysis questions based on the uploaded dataset
* Dynamically updates when a new dataset is uploaded

## Downloadable Reports

* Generate AI-assisted markdown analysis reports
* Includes dataset overview, data quality summary, AI insights, chart explanations, and recommended next steps

---

# Tech Stack

## Frontend

* Streamlit
* Matplotlib

## Backend / AI

* Python
* OpenAI API
* Pandas

## Data Processing

* Pandas
* NumPy

---

# Project Structure

```text
ai-data-analysis-workspace/
│
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
│
├── src/
│   ├── insight_agent.py
│   ├── chart_generator.py
│   ├── agent_controller.py
│   ├── quality_checks.py
│
└── README.md
```

---

# Installation

## 1. Clone the repository

```bash
git clone https://github.com/liyang6620/ai-data-analysis-workspace.git
cd ai-data-analysis-workspace
```

## 2. Create a virtual environment

```bash
python -m venv .venv
```

## 3. Activate the environment

### Windows

```bash
.venv\Scripts\activate
```

### Mac / Linux

```bash
source .venv/bin/activate
```

## 4. Install dependencies

```bash
pip install -r requirements.txt
```

## 5. Create a .env file

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

---

# Running the Application

```bash
python -m streamlit run app.py
```

---

# Example Use Cases

## Immigration and Migration Analysis

* Explore migration trends over time
* Compare visa categories
* Analyse arrivals and departures
* Identify demographic patterns

## Business Analytics

* KPI exploration
* Customer segmentation
* Trend analysis
* Category comparison

## Educational and Research Analysis

* Rapid exploratory data analysis
* AI-assisted visualisation
* Automated reporting workflows

---

# Security Notes

This project uses environment variables to protect API credentials.

The `.env` file is excluded from GitHub using `.gitignore`.

Only `.env.example` is included in the repository as a template.

---

# Future Improvements

* Interactive Plotly visualisations
* SQL database support
* PDF report export
* Multi-file analysis
* Advanced statistical analysis
* Agent memory and workflow automation
* Dashboard persistence

---

# Author

Yang Li

GitHub:
[https://github.com/liyang6620](https://github.com/liyang6620)
