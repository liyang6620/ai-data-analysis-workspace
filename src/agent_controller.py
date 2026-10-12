import pandas as pd

MAX_CONTEXT_COLUMNS = 50
MAX_CATEGORY_VALUES = 5
MAX_SAMPLE_ROWS = 5
MAX_GROUPBY_COLUMNS = 8
MAX_GROUPED_NUMERIC_COLUMNS = 8


def _compact_value(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, str):
        return value[:160]
    return value


def _grouped_numeric_summary(
    context_df: pd.DataFrame,
    categorical_columns: list[str],
    numeric_columns: list[str],
) -> dict:
    """Create bounded category-level metrics for factual AI comparisons."""

    summaries = {}
    for category in categorical_columns[:MAX_GROUPBY_COLUMNS]:
        top_values = (
            context_df[category]
            .dropna()
            .value_counts()
            .head(MAX_CATEGORY_VALUES)
            .index
        )
        category_summary = {}
        for category_value in top_values:
            category_rows = context_df[context_df[category] == category_value]
            numeric_summary = {}
            for numeric_column in numeric_columns[:MAX_GROUPED_NUMERIC_COLUMNS]:
                values = pd.to_numeric(
                    category_rows[numeric_column],
                    errors="coerce",
                ).dropna()
                if values.empty:
                    continue
                numeric_summary[numeric_column] = {
                    "count": int(values.count()),
                    "sum": round(float(values.sum()), 2),
                    "mean": round(float(values.mean()), 2),
                }
            if numeric_summary:
                category_summary[str(category_value)[:160]] = numeric_summary
        if category_summary:
            summaries[category] = category_summary

    return summaries


def prepare_dataset_context(df: pd.DataFrame) -> dict:
    """Prepare a bounded, serialisable context for the AI assistant."""
    context_columns = df.columns[:MAX_CONTEXT_COLUMNS].tolist()
    context_df = df.loc[:, context_columns]

    numeric_df = context_df.select_dtypes(include="number")
    numeric_summary = (
        numeric_df.describe().round(2).to_dict() if not numeric_df.empty else {}
    )

    categorical_summary = {}
    categorical_cols = context_df.select_dtypes(
        include=["object", "str", "category"]
    ).columns

    for col in categorical_cols:
        categorical_summary[col] = {
            str(key)[:160]: int(value)
            for key, value in context_df[col]
            .value_counts()
            .head(MAX_CATEGORY_VALUES)
            .items()
        }

    missing_values = context_df.isnull().sum()
    missing_values = missing_values[missing_values > 0].to_dict()

    sample_rows = [
        {key: _compact_value(value) for key, value in row.items()}
        for row in context_df.head(MAX_SAMPLE_ROWS).to_dict(orient="records")
    ]

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": context_columns,
        "columns_omitted": max(0, int(df.shape[1]) - len(context_columns)),
        "data_types": context_df.dtypes.astype(str).to_dict(),
        "missing_values": missing_values,
        "numeric_summary": numeric_summary,
        "top_categories": categorical_summary,
        "grouped_numeric_summary": _grouped_numeric_summary(
            context_df,
            categorical_cols.tolist(),
            numeric_df.columns.tolist(),
        ),
        "sample_rows": sample_rows
    }


def suggest_local_questions(df: pd.DataFrame) -> list[str]:
    """Offer useful prompts without making a paid model request."""
    numeric = df.select_dtypes(include="number").columns.tolist()
    categorical = df.select_dtypes(include=["object", "str", "category"]).columns.tolist()
    questions = []

    if numeric and categorical:
        questions.append(f"How does {numeric[0]} vary across {categorical[0]}?")
    if len(numeric) >= 2:
        questions.append(f"Is there a relationship between {numeric[0]} and {numeric[1]}?")
    if numeric:
        questions.append(f"Which records have the highest and lowest {numeric[0]}?")
    if categorical:
        questions.append(f"What are the most common values in {categorical[0]}?")

    return questions[:3]


def generate_auto_insights(df: pd.DataFrame) -> dict:
    """Generate simple automatic insights using local pandas rules."""

    insights = {
        "numeric_summary": {},
        "leading_categories": {},
    }

    for column in df.select_dtypes(include="number").columns[:8]:
        values = df[column].dropna()
        if values.empty:
            continue
        insights["numeric_summary"][column] = {
            "minimum": float(values.min()),
            "maximum": float(values.max()),
            "average": round(float(values.mean()), 2),
        }

    categorical_columns = df.select_dtypes(
        include=["object", "str", "category"]
    ).columns
    for column in categorical_columns[:8]:
        counts = df[column].dropna().value_counts()
        if counts.empty:
            continue
        insights["leading_categories"][column] = {
            "value": str(counts.index[0])[:160],
            "count": int(counts.iloc[0]),
        }

    if "year_month" in df.columns and "estimate" in df.columns:
        trend = df.groupby("year_month")["estimate"].sum()

        if not trend.empty:
            insights["highest_total_estimate_month"] = {
                "month": str(trend.idxmax()),
                "estimate": float(trend.max())
            }

            insights["lowest_total_estimate_month"] = {
                "month": str(trend.idxmin()),
                "estimate": float(trend.min())
            }

    if "direction" in df.columns and "estimate" in df.columns:
        direction_summary = df.groupby("direction")["estimate"].sum().sort_values(ascending=False)
        insights["estimate_by_direction"] = direction_summary.to_dict()

    if "sex" in df.columns and "estimate" in df.columns:
        sex_summary = df.groupby("sex")["estimate"].sum().sort_values(ascending=False)
        insights["estimate_by_sex"] = sex_summary.to_dict()

    if "age" in df.columns and "estimate" in df.columns:
        age_summary = df.groupby("age")["estimate"].sum().sort_values(ascending=False).head(10)
        insights["top_age_groups_by_estimate"] = age_summary.to_dict()

    return {key: value for key, value in insights.items() if value}
