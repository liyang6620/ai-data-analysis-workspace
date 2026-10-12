from pathlib import Path

import matplotlib
import streamlit as st
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.quality_checks import run_quality_checks
from src.chart_generator import (
    get_numeric_columns,
    get_categorical_columns,
    render_ai_chart,
    apply_filter,
    apply_exclusions,
    apply_numeric_filters,
    aggregate_data
)
from src.agent_controller import (
    prepare_dataset_context,
    generate_auto_insights,
    suggest_local_questions
)
from src.data_loader import DatasetLoadError, load_csv_dataset
from src.insight_agent import (
    ai_is_configured,
    answer_data_question,
    recommend_chart_spec,
    suggest_analysis_questions,
    explain_chart_result,
    generate_analysis_report
)

# PAGE CONFIG

st.set_page_config(
    page_title="AI Analysis Workspace",
    layout="wide"
)

# STYLING

st.markdown("""
<style>

.stApp {
    background: #f5f7fb;
    color: #111827;
}

.block-container {
    max-width: 1400px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

section[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e5e7eb;
}

div[role="radiogroup"] {
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
}

div[role="radiogroup"] label > div:first-child {
    display: none !important;
}

div[role="radiogroup"] label {
    position: relative;
    background: transparent;
    border: 1px solid transparent;
    border-radius: 14px;
    padding: 0.78rem 0.95rem 0.78rem 1rem;
    color: #475569;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.18s ease;
}

div[role="radiogroup"] label:hover {
    background: #f8fafc;
    color: #0f172a;
}

div[role="radiogroup"] label:has(input:checked) {
    background: linear-gradient(135deg, #eef4ff, #ffffff);
    border: 1px solid #dbeafe;
    color: #1d4ed8;
    box-shadow: 0 8px 20px rgba(37, 99, 235, 0.10);
}

div[role="radiogroup"] label:has(input:checked)::before {
    content: "";
    position: absolute;
    left: -0.55rem;
    top: 20%;
    width: 4px;
    height: 60%;
    border-radius: 999px;
    background: linear-gradient(180deg, #2563eb, #60a5fa);
}

[data-testid="metric-container"] {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 18px;
    padding: 1rem;
    box-shadow: 0 8px 22px rgba(15, 23, 42, 0.06);
}

div.stButton > button {
    height: 3rem;
    border-radius: 12px;
    font-weight: 700;
}

[data-testid="stFileUploader"] {
    background-color: #f8fafc;
    border: 1px dashed #cbd5e1;
    border-radius: 16px;
    padding: 1rem;
}

.stTextArea textarea,
.stTextInput input {
    border-radius: 12px !important;
    border: 1px solid #cbd5e1 !important;
    background-color: #ffffff !important;
    color: #111827 !important;
}

.stSelectbox div[data-baseweb="select"] {
    border-radius: 12px !important;
    background-color: #ffffff !important;
}

[data-testid="stDataFrame"] {
    border-radius: 16px;
    overflow: hidden;
    border: 1px solid #e5e7eb;
}

.panel {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 22px;
    padding: 1.4rem 1.5rem;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
    margin-bottom: 1.2rem;
}

.panel-title {
    font-size: 1.15rem;
    font-weight: 750;
    color: #0f172a;
    margin-bottom: 0.35rem;
}

.panel-text {
    color: #64748b;
    font-size: 0.95rem;
    line-height: 1.6;
}

.main-title {
    font-size: 2.6rem;
    font-weight: 850;
    color: #0f172a;
    margin-bottom: 0.3rem;
}

.main-subtitle {
    color: #64748b;
    font-size: 1.02rem;
    margin-bottom: 1.6rem;
}

h1, h2, h3 {
    color: #0f172a;
}

.stSuccess,
.stInfo,
.stWarning,
.stError {
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)

# PAGES

pages = [
    "Overview",
    "Data Quality",
    "Insights",
    "Visualisation",
    "AI Workspace"
]

# SESSION STATE

if "page_index" not in st.session_state:
    st.session_state.page_index = 0

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "ai_tool" not in st.session_state:
    st.session_state.ai_tool = "assistant"

if "suggested_questions" not in st.session_state:
    st.session_state.suggested_questions = []

if "latest_ai_answer" not in st.session_state:
    st.session_state.latest_ai_answer = ""

if "latest_chart_spec" not in st.session_state:
    st.session_state.latest_chart_spec = {}

if "latest_chart_explanation" not in st.session_state:
    st.session_state.latest_chart_explanation = ""

if "latest_report" not in st.session_state:
    st.session_state.latest_report = ""

if "chart_question" not in st.session_state:
    st.session_state.chart_question = ""

if "dataset_fingerprint" not in st.session_state:
    st.session_state.dataset_fingerprint = None

# HELPERS

def sync_page_from_radio():
    st.session_state.page_index = pages.index(
        st.session_state.main_navigation_radio
    )


def go_previous():
    if st.session_state.page_index > 0:
        st.session_state.page_index -= 1
        st.session_state.main_navigation_radio = pages[
            st.session_state.page_index
        ]


def go_next():
    if st.session_state.page_index < len(pages) - 1:
        st.session_state.page_index += 1
        st.session_state.main_navigation_radio = pages[
            st.session_state.page_index
        ]


def set_ai_tool(tool):
    st.session_state.ai_tool = tool


def get_placeholder(default_examples):
    """Use AI-suggested questions as placeholder examples when available."""
    if st.session_state.suggested_questions:
        return "\n".join(
            [f"Example: {question}" for question in st.session_state.suggested_questions]
        )

    return default_examples


def get_chart_data_summary(df, chart_spec):
    """Create a small aggregated data summary for chart explanation."""

    try:
        filtered_df = apply_filter(df, chart_spec)
        filtered_df = apply_exclusions(filtered_df, chart_spec)
        filtered_df = apply_numeric_filters(filtered_df, chart_spec)

        if chart_spec.get("chart_type") == "scatter":
            x = chart_spec.get("x")
            y = chart_spec.get("y")
            if x in filtered_df.columns and y in filtered_df.columns:
                preview = filtered_df[[x, y]].dropna().head(20)
                return {
                    "rows": int(filtered_df.shape[0]),
                    "columns": [x, y],
                    "preview": preview.to_dict(orient="records")
                }

        chart_data = aggregate_data(filtered_df, chart_spec)

        return {
            "rows": int(chart_data.shape[0]),
            "columns": chart_data.columns.tolist(),
            "preview": chart_data.head(20).to_dict(orient="records")
        }

    except Exception as e:
        return {
            "error": str(e)
        }

# SIDEBAR

with st.sidebar:

    st.header("AI Analysis Workspace")

    data_source = st.radio(
        "Data source",
        ["Demo dataset", "Upload CSV"],
        help="The demo dataset works without an API key."
    )

    if data_source == "Upload CSV":
        dataset_source = st.file_uploader(
            "Upload CSV file",
            type=["csv"],
            help="UTF-8 compatible CSV, up to 10 MB."
        )
        dataset_label = dataset_source.name if dataset_source else None
    else:
        dataset_source = Path(__file__).parent / "data" / "demo_sales.csv"
        dataset_label = "Demo retail sales"

    st.divider()

    page = st.radio(
        "Navigation",
        pages,
        index=st.session_state.page_index,
        key="main_navigation_radio",
        on_change=sync_page_from_radio
    )

    page = pages[st.session_state.page_index]

# HEADER

st.markdown(
    '<div class="main-title">AI-Powered Data Analysis Platform</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-subtitle">A modern workspace for dataset exploration, data quality review, automatic insights, AI-assisted analysis, and AI-recommended visualisation.</div>',
    unsafe_allow_html=True
)

# MAIN APP

if dataset_source is not None:

    try:
        df, dataset_fingerprint = load_csv_dataset(dataset_source)
    except DatasetLoadError as exc:
        st.error(str(exc))
        st.info("Choose the demo dataset or upload a valid CSV to continue.")
        st.stop()

    if st.session_state.dataset_fingerprint != dataset_fingerprint:
        st.session_state.dataset_fingerprint = dataset_fingerprint
        st.session_state.suggested_questions = suggest_local_questions(df)
        st.session_state.chat_history = []
        st.session_state.latest_ai_answer = ""
        st.session_state.latest_chart_spec = {}
        st.session_state.latest_chart_explanation = ""
        st.session_state.latest_report = ""
        st.session_state.chart_question = ""

    numeric_columns = get_numeric_columns(df)
    categorical_columns = get_categorical_columns(df)

    quality_summary = run_quality_checks(df)
    auto_insights = generate_auto_insights(df)

    missing_cells = int(df.isnull().sum().sum())
    duplicate_rows = int(df.duplicated().sum())

    dataset_context = prepare_dataset_context(df)

    st.caption(f"Dataset: {dataset_label}")

    # OVERVIEW

    if page == "Overview":

        st.header("Dataset Overview")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Rows", df.shape[0])

        with col2:
            st.metric("Columns", df.shape[1])

        with col3:
            st.metric("Numeric Columns", len(numeric_columns))

        with col4:
            st.metric("Categorical Columns", len(categorical_columns))

        left_col, right_col = st.columns([1.4, 1])

        with left_col:

            st.markdown("""
            <div class="panel">
                <div class="panel-title">Dataset Preview</div>
                <div class="panel-text">
                    Preview the first rows to confirm that the uploaded dataset has been loaded correctly.
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.dataframe(df.head(), width="stretch")

        with right_col:

            st.markdown("""
            <div class="panel">
                <div class="panel-title">Column Profile</div>
                <div class="panel-text">
                    Review column names, data types, missing values, and missing rate.
                </div>
            </div>
            """, unsafe_allow_html=True)

            column_info = pd.DataFrame({
                "Column": df.columns,
                "Data Type": df.dtypes.astype(str),
                "Missing Values": df.isnull().sum().values,
                "Missing Rate (%)": (df.isnull().mean().values * 100).round(2)
            })

            st.dataframe(column_info, width="stretch")

    # DATA QUALITY

    elif page == "Data Quality":

        st.header("Data Quality Review")

        q1, q2, q3 = st.columns(3)

        with q1:
            st.metric("Missing Cells", missing_cells)

        with q2:
            st.metric("Duplicate Rows", duplicate_rows)

        with q3:
            st.metric(
                "Constant Columns",
                len(quality_summary["constant_columns"])
            )

        st.markdown("""
        <div class="panel">
            <div class="panel-title">Quality Assessment</div>
            <div class="panel-text">
                This section checks missing values, duplicated records, and constant columns that may affect analysis quality.
            </div>
        </div>
        """, unsafe_allow_html=True)

        missing_by_column = quality_summary["missing_values"]
        constant_columns = quality_summary["constant_columns"]

        if not missing_by_column and not constant_columns and duplicate_rows == 0:
            st.success("No missing values, duplicate rows, or constant columns detected.")
        else:
            quality_left, quality_right = st.columns(2)
            with quality_left:
                st.subheader("Missing Values")
                if missing_by_column:
                    missing_table = pd.DataFrame(
                        [
                            {
                                "Column": column,
                                "Missing Cells": count,
                                "Missing Rate (%)": round(count / len(df) * 100, 2),
                            }
                            for column, count in missing_by_column.items()
                        ]
                    )
                    st.dataframe(missing_table, width="stretch", hide_index=True)
                else:
                    st.caption("No missing values detected.")

            with quality_right:
                st.subheader("Structural Checks")
                st.write(f"Duplicate rows: **{duplicate_rows}**")
                if constant_columns:
                    st.write("Constant columns: " + ", ".join(constant_columns))
                else:
                    st.caption("No constant columns detected.")

    # INSIGHTS

    elif page == "Insights":

        st.header("Automatic Data Insights")

        st.markdown("""
        <div class="panel">
            <div class="panel-title">Insight Engine</div>
            <div class="panel-text">
                This module applies automatic rules to highlight useful dataset findings before deeper analysis.
            </div>
        </div>
        """, unsafe_allow_html=True)

        numeric_insights = auto_insights.get("numeric_summary", {})
        category_insights = auto_insights.get("leading_categories", {})

        insight_left, insight_right = st.columns(2)
        with insight_left:
            st.subheader("Numeric Ranges")
            if numeric_insights:
                numeric_table = pd.DataFrame(
                    [
                        {
                            "Measure": column,
                            "Minimum": values["minimum"],
                            "Average": values["average"],
                            "Maximum": values["maximum"],
                        }
                        for column, values in numeric_insights.items()
                    ]
                )
                st.dataframe(numeric_table, width="stretch", hide_index=True)
            else:
                st.caption("No numeric columns are available.")

        with insight_right:
            st.subheader("Leading Categories")
            if category_insights:
                category_table = pd.DataFrame(
                    [
                        {
                            "Column": column,
                            "Most Common Value": values["value"],
                            "Records": values["count"],
                        }
                        for column, values in category_insights.items()
                    ]
                )
                st.dataframe(category_table, width="stretch", hide_index=True)
            else:
                st.caption("No categorical columns are available.")

        domain_insights = {
            key: value
            for key, value in auto_insights.items()
            if key not in {"numeric_summary", "leading_categories"}
        }
        if domain_insights:
            with st.expander("Dataset-specific signals"):
                st.json(domain_insights)

        st.subheader("Suggested Analysis Questions")

        if st.session_state.suggested_questions:
            for i, question in enumerate(st.session_state.suggested_questions, start=1):
                st.write(f"{i}. {question}")
        else:
            st.info("No suggested questions available yet.")

        if st.button(
            "Refine questions with AI",
            disabled=not ai_is_configured(),
            help=(
                None
                if ai_is_configured()
                else "Configure OPENAI_API_KEY to enable AI-generated questions."
            ),
        ):
            try:
                with st.spinner("Refining questions..."):
                    st.session_state.suggested_questions = suggest_analysis_questions(
                        dataset_context
                    )
                st.rerun()
            except Exception:
                st.error("The AI service could not generate questions. Try again later.")

    # VISUALISATION

    elif page == "Visualisation":

        st.header("Basic Data Visualisation")

        chart_type = st.selectbox(
            "Select visualisation type",
            [
                "Numeric Distribution",
                "Categorical Distribution",
                "Scatter Plot"
            ],
            key="chart_type_select"
        )

        if chart_type == "Numeric Distribution":

            if numeric_columns:

                selected_numeric = st.selectbox(
                    "Select a numeric column",
                    numeric_columns,
                    key="numeric_hist"
                )

                fig, ax = plt.subplots(figsize=(8, 4.5))

                ax.hist(
                    df[selected_numeric].dropna(),
                    bins=20
                )

                ax.set_title(
                    f"Distribution of {selected_numeric}"
                )

                ax.grid(alpha=0.25)

                st.pyplot(fig)

        elif chart_type == "Categorical Distribution":

            if categorical_columns:

                selected_category = st.selectbox(
                    "Select a categorical column",
                    categorical_columns,
                    key="category_bar"
                )

                value_counts = (
                    df[selected_category]
                    .value_counts()
                    .head(10)
                )

                fig, ax = plt.subplots(figsize=(8, 4.5))

                ax.bar(
                    value_counts.index.astype(str),
                    value_counts.values
                )

                ax.set_title(
                    f"Top Categories in {selected_category}"
                )

                plt.xticks(rotation=45)

                st.pyplot(fig)

        else:

            if len(numeric_columns) >= 2:

                c1, c2 = st.columns(2)

                with c1:
                    x_col = st.selectbox(
                        "X Axis",
                        numeric_columns,
                        key="scatter_x"
                    )

                with c2:
                    y_col = st.selectbox(
                        "Y Axis",
                        numeric_columns,
                        key="scatter_y"
                    )

                fig, ax = plt.subplots(figsize=(8, 4.5))

                ax.scatter(
                    df[x_col],
                    df[y_col],
                    alpha=0.7
                )

                ax.set_title(
                    f"{x_col} vs {y_col}"
                )

                ax.grid(alpha=0.25)

                st.pyplot(fig)

    # AI WORKSPACE

    elif page == "AI Workspace":

        st.header("AI Workspace")

        ai_ready = ai_is_configured()
        if not ai_ready:
            st.info(
                "AI features are optional and currently unavailable because no API key "
                "is configured. Dataset profiling and manual charts remain fully available."
            )

        st.markdown("""
        <div class="panel">
            <div class="panel-title">AI Analysis Module</div>
            <div class="panel-text">
                Use AI tools to ask analytical questions or automatically generate charts from natural language.
            </div>
        </div>
        """, unsafe_allow_html=True)

        tool_col1, tool_col2 = st.columns(2)

        with tool_col1:

            assistant_style = (
                "primary"
                if st.session_state.ai_tool == "assistant"
                else "secondary"
            )

            st.button(
                "Ask AI Assistant",
                width="stretch",
                type=assistant_style,
                key="ai_tool_selector_assistant",
                on_click=set_ai_tool,
                args=("assistant",)
            )

        with tool_col2:

            chart_style = (
                "primary"
                if st.session_state.ai_tool == "chart"
                else "secondary"
            )

            st.button(
                "Generate AI Chart",
                width="stretch",
                type=chart_style,
                key="ai_tool_selector_chart",
                on_click=set_ai_tool,
                args=("chart",)
            )

        st.markdown("<br>", unsafe_allow_html=True)

        if st.session_state.ai_tool == "assistant":

            st.markdown("""
            <div class="panel">
                <div class="panel-title">Ask Analytical Questions</div>
                <div class="panel-text">
                    Request dataset explanations, Python guidance, analytical interpretation, or business insights.
                </div>
            </div>
            """, unsafe_allow_html=True)

            user_question = st.text_area(
                "Your question",
                height=180,
                placeholder=get_placeholder(
                    "Example: How can I find the highest estimate in 2010?\n"
                    "Example: Give me Python code to plot arrivals and departures by month.\n"
                    "Example: How can I calculate net migration using this dataset?"
                ),
                key="assistant_question"
            )

            if st.button(
                "Ask AI Assistant",
                width="stretch",
                key="assistant_submit",
                disabled=not ai_ready
            ):

                if not user_question.strip():

                    st.warning("Please enter a question first.")

                else:

                    try:

                        with st.spinner("Generating answer..."):

                            answer = answer_data_question(
                                dataset_context,
                                user_question,
                                st.session_state.chat_history
                            )

                        st.subheader("AI Answer")
                        st.write(answer)

                        st.session_state.latest_ai_answer = answer

                        st.session_state.chat_history.append({
                            "question": user_question,
                            "answer": answer
                        })

                    except Exception:
                        st.error("AI response failed.")
                        st.caption("Check the API configuration and try again.")

        else:

            st.markdown("""
            <div class="panel">
                <div class="panel-title">Generate AI Visualisations</div>
                <div class="panel-text">
                    Describe the chart you want in natural language and let the AI recommend and generate it automatically.
                </div>
            </div>
            """, unsafe_allow_html=True)

            chart_question = st.text_area(
                "Describe the chart you want",
                height=180,
                placeholder=get_placeholder(
                    "Example: Show the distribution of estimate by visa using a boxplot, excluding totals and only include estimate values under 2000.\n"
                    "Example: Show which visa categories contribute the most, split by citizenship, excluding totals.\n"
                    "Example: Show migration trends over time."
                ),
                key="chart_question"
            )

            if st.button(
                "Generate AI Chart",
                width="stretch",
                key="chart_submit",
                disabled=not ai_ready
            ):

                if not chart_question.strip():

                    st.warning("Please describe the chart you want first.")

                else:

                    try:

                        with st.spinner("Generating chart recommendation..."):

                            chart_spec = recommend_chart_spec(
                                dataset_context,
                                chart_question
                            )

                        st.subheader("AI Chart Specification")
                        st.json(chart_spec)

                        fig = render_ai_chart(
                            df,
                            chart_spec
                        )

                        st.pyplot(fig)

                        st.session_state.latest_chart_spec = chart_spec

                        chart_data_summary = get_chart_data_summary(
                            df,
                            chart_spec
                        )

                        with st.spinner("Explaining chart result..."):
                            chart_explanation = explain_chart_result(
                                dataset_context,
                                chart_spec,
                                chart_data_summary
                            )

                        st.subheader("AI Chart Explanation")
                        st.write(chart_explanation)

                        st.session_state.latest_chart_explanation = chart_explanation

                    except Exception:
                        st.error("Chart generation failed.")
                        st.caption("Check the request, dataset columns, and API configuration.")

        st.markdown("<br><br>", unsafe_allow_html=True)

        st.header("Conversation History")

        if st.session_state.chat_history:

            for i, item in enumerate(
                st.session_state.chat_history,
                start=1
            ):

                with st.expander(
                    f"Question {i}: {item['question']}"
                ):

                    st.write(item["answer"])

        else:

            st.info("No questions asked yet.")

        st.divider()

        st.header("Download Analysis Report")

        if st.button(
            "Generate Report",
            width="stretch",
            key="generate_report_button",
            disabled=not ai_ready
        ):
            report_context = {
                "dataset_overview": {
                    "rows": int(df.shape[0]),
                    "columns": int(df.shape[1]),
                    "numeric_columns": len(numeric_columns),
                    "categorical_columns": len(categorical_columns)
                },
                "data_quality": quality_summary,
                "automatic_insights": auto_insights,
                "suggested_questions": st.session_state.suggested_questions,
                "latest_ai_answer": st.session_state.latest_ai_answer,
                "latest_chart_spec": st.session_state.latest_chart_spec,
                "latest_chart_explanation": st.session_state.latest_chart_explanation
            }

            try:
                with st.spinner("Generating report..."):
                    st.session_state.latest_report = generate_analysis_report(
                        report_context
                    )

            except Exception:
                st.error("Report generation failed.")
                st.caption("Check the API configuration and try again.")

        if st.session_state.latest_report:
            st.download_button(
                label="Download Report as Markdown",
                data=st.session_state.latest_report,
                file_name="ai_analysis_report.md",
                mime="text/markdown",
                width="stretch"
            )

    # NAVIGATION BUTTONS

    st.divider()

    previous_col, spacer_col, next_col = st.columns([1, 4, 1])

    with previous_col:

        st.button(
            "Previous",
            width="stretch",
            disabled=st.session_state.page_index == 0,
            on_click=go_previous,
            key="previous_button"
        )

    with next_col:

        st.button(
            "Next",
            width="stretch",
            disabled=st.session_state.page_index == len(pages) - 1,
            on_click=go_next,
            key="next_button"
        )

else:

    st.info(
        "Upload a CSV file from the sidebar or choose the demo dataset to begin."
    )
