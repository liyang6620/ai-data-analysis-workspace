import pandas as pd
import matplotlib.pyplot as plt


def get_numeric_columns(df: pd.DataFrame) -> list:
    """Return numeric columns from the dataset."""
    return df.select_dtypes(include=["int64", "float64"]).columns.tolist()


def get_categorical_columns(df: pd.DataFrame) -> list:
    """Return categorical columns from the dataset."""
    return df.select_dtypes(include=["object", "category"]).columns.tolist()


def apply_filter(df: pd.DataFrame, chart_spec: dict) -> pd.DataFrame:
    """Apply a simple text filter to the dataset based on the chart specification."""

    filter_column = chart_spec.get("filter_column")
    filter_operator = chart_spec.get("filter_operator")
    filter_value = chart_spec.get("filter_value")

    if not filter_column or filter_column not in df.columns:
        return df

    filtered_df = df.copy()

    col = filtered_df[filter_column].astype(str).str.strip().str.lower()
    val = str(filter_value).strip().lower()

    if filter_operator == "startswith":
        mask = col.str.startswith(val)
    elif filter_operator == "equals":
        mask = col == val
    elif filter_operator == "notequals":
        mask = col != val
    elif filter_operator == "contains":
        mask = col.str.contains(val, na=False)
    elif filter_operator == "notcontains":
        mask = ~col.str.contains(val, na=False)
    else:
        return df

    return filtered_df[mask]


def apply_exclusions(df: pd.DataFrame, chart_spec: dict) -> pd.DataFrame:
    """Apply flexible value exclusions across multiple columns."""

    exclude_values = chart_spec.get("exclude_values", {})

    if not isinstance(exclude_values, dict):
        return df

    cleaned_df = df.copy()

    for col, values in exclude_values.items():
        if col not in cleaned_df.columns:
            continue

        if not isinstance(values, list):
            values = [values]

        normalised_values = [
            str(value).strip().lower()
            for value in values
        ]

        col_values = cleaned_df[col].astype(str).str.strip().str.lower()

        cleaned_df = cleaned_df[
            ~col_values.isin(normalised_values)
        ]

    return cleaned_df


def apply_numeric_filters(df: pd.DataFrame, chart_spec: dict) -> pd.DataFrame:
    """Apply numeric filters such as less than, greater than, or between."""

    numeric_filters = chart_spec.get("numeric_filters", [])

    if not isinstance(numeric_filters, list):
        return df

    filtered_df = df.copy()

    for rule in numeric_filters:
        column = rule.get("column")
        operator = rule.get("operator")
        value = rule.get("value")

        if column not in filtered_df.columns:
            continue

        filtered_df[column] = pd.to_numeric(filtered_df[column], errors="coerce")

        if operator == "lt":
            filtered_df = filtered_df[filtered_df[column] < value]
        elif operator == "lte":
            filtered_df = filtered_df[filtered_df[column] <= value]
        elif operator == "gt":
            filtered_df = filtered_df[filtered_df[column] > value]
        elif operator == "gte":
            filtered_df = filtered_df[filtered_df[column] >= value]
        elif operator == "between":
            if isinstance(value, list) and len(value) == 2:
                filtered_df = filtered_df[
                    (filtered_df[column] >= value[0]) &
                    (filtered_df[column] <= value[1])
                ]

    return filtered_df


def aggregate_data(df: pd.DataFrame, chart_spec: dict) -> pd.DataFrame:
    """Aggregate data based on the chart specification."""

    x = chart_spec.get("x")
    y = chart_spec.get("y")
    group_by = chart_spec.get("group_by")
    aggregation = chart_spec.get("aggregation", "sum")

    if x is None:
        raise ValueError("The AI did not choose an x-axis column.")

    if x not in df.columns:
        raise ValueError(f"Column '{x}' does not exist in the dataset.")

    if aggregation != "count":
        if y is None:
            raise ValueError("The AI did not choose a y-axis column.")

        if y not in df.columns:
            raise ValueError(f"Column '{y}' does not exist in the dataset.")

    group_columns = [x]

    if group_by and group_by in df.columns:
        group_columns.append(group_by)

    if aggregation == "count":
        result = df.groupby(group_columns).size().reset_index(name="value")
    else:
        result = df.groupby(group_columns)[y].agg(aggregation).reset_index()
        result = result.rename(columns={y: "value"})

    return result


def render_ai_chart(df: pd.DataFrame, chart_spec: dict):
    """Render a chart based on a safe AI-generated chart specification."""

    chart_type = chart_spec.get("chart_type")
    x = chart_spec.get("x")
    y = chart_spec.get("y")
    group_by = chart_spec.get("group_by")
    title = chart_spec.get("title", "AI Generated Chart")

    filtered_df = apply_filter(df, chart_spec)
    filtered_df = apply_exclusions(filtered_df, chart_spec)
    filtered_df = apply_numeric_filters(filtered_df, chart_spec)

    fig, ax = plt.subplots()

    if chart_type == "histogram":
        if y and y in filtered_df.columns:
            numeric_col = y
        elif x and x in filtered_df.columns:
            numeric_col = x
        else:
            raise ValueError("Histogram requires a numeric column.")

        ax.hist(filtered_df[numeric_col].dropna(), bins=20)
        ax.set_title(title)
        ax.set_xlabel(numeric_col)
        ax.set_ylabel("Frequency")
        return fig

    if chart_type == "boxplot":
        if not x or x not in filtered_df.columns:
            raise ValueError("Boxplot requires a valid categorical x column.")

        if not y or y not in filtered_df.columns:
            raise ValueError("Boxplot requires a valid numeric y column.")

        plot_df = filtered_df[[x, y]].dropna()

        grouped = plot_df.groupby(x)[y]
        groups = [group.values for _, group in grouped]
        labels = [str(label) for label in grouped.groups.keys()]

        ax.boxplot(groups, labels=labels)
        ax.set_title(title)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        return fig

    if chart_type == "heatmap":
        numeric_df = filtered_df.select_dtypes(include=["int64", "float64"])

        if numeric_df.shape[1] < 2:
            raise ValueError("Heatmap requires at least two numeric columns.")

        corr = numeric_df.corr()

        im = ax.imshow(corr)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right")
        ax.set_yticklabels(corr.columns)

        for i in range(len(corr.columns)):
            for j in range(len(corr.columns)):
                ax.text(j, i, round(corr.iloc[i, j], 2), ha="center", va="center")

        ax.set_title(title)
        fig.colorbar(im, ax=ax)
        plt.tight_layout()
        return fig

    chart_data = aggregate_data(filtered_df, chart_spec)

    if chart_type == "pie":
        if group_by:
            raise ValueError(
                "Pie chart does not support grouped data. Use grouped_bar or stacked_bar instead."
            )

        labels = chart_data[x].astype(str)
        values = chart_data["value"]

        ax.pie(values, labels=labels, autopct="%1.1f%%")
        ax.set_title(title)
        return fig

    if chart_type == "grouped_bar":
        if not group_by or group_by not in chart_data.columns:
            raise ValueError("grouped_bar requires a valid group_by column.")

        pivot_data = chart_data.pivot(
            index=x,
            columns=group_by,
            values="value"
        ).fillna(0)

        pivot_data.plot(kind="bar", ax=ax)
        ax.set_title(title)
        ax.set_xlabel(x)
        ax.set_ylabel("Value")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        return fig

    if chart_type == "stacked_bar":
        if not group_by or group_by not in chart_data.columns:
            raise ValueError("stacked_bar requires a valid group_by column.")

        pivot_data = chart_data.pivot(
            index=x,
            columns=group_by,
            values="value"
        ).fillna(0)

        pivot_data.plot(kind="bar", stacked=True, ax=ax)
        ax.set_title(title)
        ax.set_xlabel(x)
        ax.set_ylabel("Value")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        return fig

    if chart_type == "area":
        if group_by and group_by in chart_data.columns:
            pivot_data = chart_data.pivot(
                index=x,
                columns=group_by,
                values="value"
            ).fillna(0)

            pivot_data.plot(kind="area", ax=ax)
        else:
            ax.fill_between(chart_data[x].astype(str), chart_data["value"])

        ax.set_title(title)
        ax.set_xlabel(x)
        ax.set_ylabel("Value")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        return fig

    if group_by and group_by in chart_data.columns:
        for group_value in chart_data[group_by].unique():
            subset = chart_data[chart_data[group_by] == group_value]

            if chart_type == "line":
                ax.plot(
                    subset[x].astype(str),
                    subset["value"],
                    marker="o",
                    label=str(group_value)
                )
            elif chart_type == "bar":
                ax.bar(
                    subset[x].astype(str),
                    subset["value"],
                    label=str(group_value)
                )
            elif chart_type == "scatter":
                ax.scatter(
                    subset[x].astype(str),
                    subset["value"],
                    label=str(group_value)
                )
            else:
                raise ValueError(f"Unsupported chart type: {chart_type}")

        ax.legend()

    else:
        if chart_type == "line":
            ax.plot(chart_data[x].astype(str), chart_data["value"], marker="o")
        elif chart_type == "bar":
            ax.bar(chart_data[x].astype(str), chart_data["value"])
        elif chart_type == "scatter":
            ax.scatter(chart_data[x].astype(str), chart_data["value"])
        else:
            raise ValueError(f"Unsupported chart type: {chart_type}")

    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel("Value")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    return fig