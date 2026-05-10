import pandas as pd


def prepare_dataset_context(df: pd.DataFrame) -> dict:
    """Prepare a compact dataset context for the AI assistant."""

    numeric_summary = df.describe(include="number").round(2).to_dict()

    categorical_summary = {}
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns

    for col in categorical_cols:
        categorical_summary[col] = df[col].value_counts().head(10).to_dict()

    missing_values = df.isnull().sum()
    missing_values = missing_values[missing_values > 0].to_dict()

    sample_rows = df.head(10).to_dict(orient="records")

    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_names": df.columns.tolist(),
        "data_types": df.dtypes.astype(str).to_dict(),
        "missing_values": missing_values,
        "numeric_summary": numeric_summary,
        "top_categories": categorical_summary,
        "sample_rows": sample_rows
    }


def generate_auto_insights(df: pd.DataFrame) -> dict:
    """Generate simple automatic insights using local pandas rules."""

    insights = {}

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

    return insights