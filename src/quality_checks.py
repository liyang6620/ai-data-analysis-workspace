import pandas as pd


def run_quality_checks(df: pd.DataFrame) -> dict:
    """Run basic data quality checks on the uploaded dataset."""

    missing_values = df.isnull().sum()
    duplicate_rows = df.duplicated().sum()

    constant_columns = [
        col for col in df.columns
        if df[col].nunique(dropna=False) == 1
    ]

    high_missing_columns = missing_values[
        missing_values > 0
    ].sort_values(ascending=False)

    return {
        "duplicate_rows": int(duplicate_rows),
        "missing_values": high_missing_columns.to_dict(),
        "constant_columns": constant_columns
    }