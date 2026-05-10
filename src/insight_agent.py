import os
import json
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

api_key = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=api_key)


def answer_data_question(
    dataset_context: dict,
    user_question: str,
    chat_history: list
) -> str:
    """Answer a user's specific data analysis question using OpenAI."""

    prompt = f"""
You are a practical data analysis assistant.

The user has uploaded a CSV dataset. You are given a compact summary of the dataset.

Your job is to answer the user's specific question.

Important rules:
- Do not give generic business advice.
- Focus only on the user's question.
- Use the actual column names from the dataset.
- If the user asks how to calculate something, explain the exact columns and filtering logic.
- If the user asks for Python code, provide pandas/matplotlib code.
- If the user asks for R code, provide tidyverse/ggplot2 code.
- If the dataset does not contain the required column, clearly say what is missing.
- Do not invent columns that are not in the dataset.
- Keep the answer practical and directly usable.
- Use the conversation history only when it helps answer follow-up questions.

Conversation history:
{chat_history}

Dataset context:
{dataset_context}

User question:
{user_question}
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    return response.output_text


def recommend_chart_spec(dataset_context: dict, chart_question: str) -> dict:
    """Ask OpenAI to recommend a safe chart specification."""

    prompt = f"""
You are an AI data visualisation agent.

The user has uploaded a CSV dataset and wants to visualise something.

Return ONLY a valid JSON object. Do not include markdown or explanation.

Chart type selection rules:
- If the user explicitly requests a supported chart type, respect that chart type whenever possible.
- If the requested chart type is incompatible with the selected data structure, choose the closest suitable alternative.
- If the user does not specify a chart type, choose the most suitable chart type based on the analytical goal.
- Use line charts for time trends.
- Use bar charts for category comparisons.
- Use grouped_bar charts for category comparisons across groups.
- Use stacked_bar charts for part-to-whole comparisons across categories and groups.
- Use scatter charts for relationships between two numeric variables.
- Use pie charts only for simple part-to-whole composition with one categorical variable and one numeric measure.
- Use histogram for numeric distributions.
- Use boxplot for comparing the distribution of a numeric variable across categories.
- Use area charts for cumulative or time-based volume comparison.
- Use heatmap for correlation or matrix-style comparison.
- If the user asks for grouped data in a pie chart, choose grouped_bar or stacked_bar instead.

Supported chart types:
- line
- bar
- grouped_bar
- stacked_bar
- scatter
- pie
- histogram
- boxplot
- area
- heatmap

Supported aggregation methods:
- sum
- mean
- count

Filtering and exclusion rules:
- Use only columns that exist in the dataset.
- Do not invent columns.
- If the user asks to remove totals, unknowns, aggregates, missing categories, or irrelevant categories, put them in exclude_values.
- exclude_values must be a dictionary where each key is a column name and each value is a list of values to exclude.
- If the user asks for numeric filtering such as under, below, less than, greater than, over, at least, at most, or between, use numeric_filters.
- Supported numeric filter operators: lt, lte, gt, gte, between.
- Use null if no filter is needed.
- Always return a valid chart specification.
- Never return an error field.

The JSON must follow this structure:

{{
  "chart_type": "boxplot",
  "x": "visa",
  "y": "estimate",
  "group_by": null,
  "aggregation": "sum",
  "filter_column": null,
  "filter_operator": null,
  "filter_value": null,
  "exclude_values": {{
    "visa": ["TOTAL"],
    "citizenship": ["TOTAL"]
  }},
  "numeric_filters": [
    {{
      "column": "estimate",
      "operator": "lt",
      "value": 2000
    }}
  ],
  "title": "Distribution of Estimate by Visa Under 2000"
}}

Dataset context:
{dataset_context}

User chart request:
{chart_question}
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    raw_text = response.output_text.strip()

    if raw_text.startswith("```"):
        raw_text = raw_text.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        raise ValueError(f"AI did not return valid JSON: {raw_text}")


def suggest_analysis_questions(dataset_context: dict) -> list:
    """Suggest three useful analysis questions for the uploaded dataset."""

    prompt = f"""
You are a data analytics product assistant.

Based on the uploaded dataset context, suggest exactly 3 specific and useful analysis questions.

Rules:
- Questions must be specific to the actual columns in the dataset.
- Do not give generic questions.
- Prefer questions that can lead to charts, KPIs, or business decisions.
- Return ONLY a valid JSON list of strings.
- Do not include markdown.

Dataset context:
{dataset_context}
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    raw_text = response.output_text.strip()

    if raw_text.startswith("```"):
        raw_text = raw_text.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        raise ValueError(f"AI did not return valid JSON: {raw_text}")


def explain_chart_result(
    dataset_context: dict,
    chart_spec: dict,
    chart_data_summary: dict
) -> str:
    """Explain an AI-generated chart result in practical language."""

    prompt = f"""
You are a practical data analyst.

Explain the chart result based on:
1. The dataset context
2. The chart specification
3. The aggregated chart data summary

Rules:
- Do not describe the chart generically.
- Explain what the chart is showing.
- Mention the most important categories, values, trends, or comparisons.
- Explain what a business user or analyst can learn from it.
- Mention any limitation if the chart cannot support strong conclusions.
- Keep the explanation concise and useful.

Dataset context:
{dataset_context}

Chart specification:
{chart_spec}

Aggregated chart data summary:
{chart_data_summary}
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    return response.output_text


def generate_analysis_report(report_context: dict) -> str:
    """Generate a downloadable markdown analysis report."""

    prompt = f"""
You are a professional data analyst.

Generate a concise markdown report based on the following analysis context.

The report should include:
1. Dataset Overview
2. Data Quality Summary
3. Automatic Insights
4. AI Question Answer Summary
5. AI Chart Summary
6. Recommended Next Steps

Rules:
- Use clear headings.
- Be practical and concise.
- Do not invent results.
- If a section has no content, say it has not been generated yet.

Report context:
{report_context}
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    return response.output_text
