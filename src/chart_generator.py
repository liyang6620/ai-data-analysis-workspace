import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ALLOWED_CHART_TYPES = {
    "line",
    "bar",
    "grouped_bar",
    "stacked_bar",
    "scatter",
    "pie",
    "histogram",
    "boxplot",
    "area",
    "heatmap",
}
ALLOWED_AGGREGATIONS = {"sum", "mean", "count"}
ALLOWED_TEXT_FILTERS = {
    "startswith",
    "equals",
    "notequals",
    "contains",
    "notcontains",
    "in",
    "notin",
}
ALLOWED_NUMERIC_FILTERS = {"lt", "lte", "gt", "gte", "between"}
ALLOWED_SORT_ORDERS = {"auto", "ascending", "descending", "chronological"}
MAX_CHART_TITLE_LENGTH = 120
MAX_FILTER_RULES = 10
MAX_TEXT_FILTER_VALUES = 100


def get_numeric_columns(df: pd.DataFrame) -> list:
    """Return numeric columns from the dataset."""
    return df.select_dtypes(include="number").columns.tolist()


def get_categorical_columns(df: pd.DataFrame) -> list:
    """Return categorical columns from the dataset."""
    return df.select_dtypes(include=["object", "str", "category"]).columns.tolist()


def apply_filter(df: pd.DataFrame, chart_spec: dict) -> pd.DataFrame:
    """Apply legacy and multi-rule text filters to the dataset."""

    text_filters = chart_spec.get("text_filters")
    if not isinstance(text_filters, list):
        text_filters = []

    filter_column = chart_spec.get("filter_column")
    if filter_column:
        text_filters = [
            {
                "column": filter_column,
                "operator": chart_spec.get("filter_operator"),
                "value": chart_spec.get("filter_value"),
            },
            *text_filters,
        ]

    filtered_df = df.copy()
    for rule in text_filters:
        if not isinstance(rule, dict):
            continue

        column = rule.get("column")
        operator = rule.get("operator")
        value = rule.get("value")
        if not column or column not in filtered_df.columns:
            continue

        col = filtered_df[column].astype(str).str.strip().str.lower()
        values = value if isinstance(value, list) else [value]
        normalised_values = [str(item).strip().lower() for item in values]

        if operator == "startswith":
            mask = col.str.startswith(normalised_values[0])
        elif operator == "equals":
            mask = col == normalised_values[0]
        elif operator == "notequals":
            mask = col != normalised_values[0]
        elif operator == "contains":
            mask = col.str.contains(normalised_values[0], na=False, regex=False)
        elif operator == "notcontains":
            mask = ~col.str.contains(normalised_values[0], na=False, regex=False)
        elif operator == "in":
            mask = col.isin(normalised_values)
        elif operator == "notin":
            mask = ~col.isin(normalised_values)
        else:
            continue

        filtered_df = filtered_df[mask]

    return filtered_df


def _validate_text_filter(df: pd.DataFrame, rule: dict) -> dict:
    """Validate and normalise one text-filter rule."""

    if not isinstance(rule, dict):
        raise TypeError("Each text filter must be an object.")

    normalised = dict(rule)
    column = _require_column(df, normalised.get("column"), "text filter")
    operator = str(normalised.get("operator", "")).strip().lower()
    operator = {"one_of": "in", "not_in": "notin"}.get(operator, operator)
    if operator not in ALLOWED_TEXT_FILTERS:
        raise ValueError("The text filter operator is not supported.")

    value = normalised.get("value")
    if value is None or isinstance(value, dict):
        raise ValueError("A text filter requires a filter value.")

    values = value if isinstance(value, list) else [value]
    if not values or any(item is None for item in values):
        raise ValueError("A text filter requires at least one filter value.")
    if len(values) > MAX_TEXT_FILTER_VALUES:
        raise ValueError(
            f"A text filter can contain at most {MAX_TEXT_FILTER_VALUES} values."
        )
    if operator in {"in", "notin"}:
        normalised["value"] = values
    elif len(values) != 1:
        raise ValueError(f"The {operator} text filter accepts one value only.")
    else:
        normalised["value"] = values[0]

    normalised["column"] = column
    normalised["operator"] = operator
    return normalised


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

        if operator == "between":
            if not isinstance(value, list) or len(value) != 2:
                continue
            bounds = pd.to_numeric(pd.Series(value), errors="coerce")
            if bounds.isna().any():
                continue
            filtered_df = filtered_df[
                (filtered_df[column] >= bounds.iloc[0])
                & (filtered_df[column] <= bounds.iloc[1])
            ]
            continue

        numeric_value = pd.to_numeric(value, errors="coerce")
        if pd.isna(numeric_value):
            continue
        if operator == "lt":
            filtered_df = filtered_df[filtered_df[column] < numeric_value]
        elif operator == "lte":
            filtered_df = filtered_df[filtered_df[column] <= numeric_value]
        elif operator == "gt":
            filtered_df = filtered_df[filtered_df[column] > numeric_value]
        elif operator == "gte":
            filtered_df = filtered_df[filtered_df[column] >= numeric_value]

    return filtered_df


def _require_column(df: pd.DataFrame, column: object, role: str) -> str:
    if not isinstance(column, str) or not column:
        raise ValueError(f"The chart requires a {role} column.")
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist in the dataset.")
    return column


def _require_numeric_data(df: pd.DataFrame, column: str, role: str) -> None:
    values = pd.to_numeric(df[column], errors="coerce")
    if not values.notna().any():
        raise ValueError(f"The {role} column '{column}' must contain numeric values.")


def validate_chart_spec(df: pd.DataFrame, chart_spec: dict) -> dict:
    """Validate and normalise the constrained chart specification."""

    if not isinstance(chart_spec, dict):
        raise TypeError("The AI chart specification must be a JSON object.")

    spec = dict(chart_spec)
    chart_type = str(spec.get("chart_type", "")).strip().lower()
    if chart_type not in ALLOWED_CHART_TYPES:
        supported = ", ".join(sorted(ALLOWED_CHART_TYPES))
        raise ValueError(f"Unsupported chart type. Choose one of: {supported}.")
    spec["chart_type"] = chart_type

    aggregation = str(spec.get("aggregation", "sum")).strip().lower()
    if aggregation not in ALLOWED_AGGREGATIONS:
        raise ValueError("Aggregation must be sum, mean, or count.")
    spec["aggregation"] = aggregation

    title = spec.get("title") or "AI Generated Chart"
    if not isinstance(title, str):
        raise TypeError("The chart title must be text.")
    title = title.strip() or "AI Generated Chart"
    if len(title) > MAX_CHART_TITLE_LENGTH:
        raise ValueError(
            f"The chart title must be {MAX_CHART_TITLE_LENGTH} characters or fewer."
        )
    spec["title"] = title

    x = spec.get("x")
    y = spec.get("y")
    group_by = spec.get("group_by")

    if x is not None:
        _require_column(df, x, "x-axis")
    if y is not None:
        _require_column(df, y, "y-axis")
    if group_by is not None:
        _require_column(df, group_by, "grouping")

    if chart_type == "heatmap":
        if len(get_numeric_columns(df)) < 2:
            raise ValueError("A heatmap requires at least two numeric columns.")
    elif chart_type == "histogram":
        numeric_column = y or x
        numeric_column = _require_column(df, numeric_column, "numeric")
        _require_numeric_data(df, numeric_column, "histogram")
    elif chart_type == "scatter":
        x = _require_column(df, x, "x-axis")
        y = _require_column(df, y, "y-axis")
        _require_numeric_data(df, x, "x-axis")
        _require_numeric_data(df, y, "y-axis")
    elif chart_type == "boxplot":
        _require_column(df, x, "category")
        y = _require_column(df, y, "numeric")
        _require_numeric_data(df, y, "boxplot")
    else:
        _require_column(df, x, "x-axis")
        if aggregation != "count":
            y = _require_column(df, y, "y-axis")
            _require_numeric_data(df, y, "y-axis")

    grouped_types = {"grouped_bar", "stacked_bar"}
    if chart_type in grouped_types and not group_by:
        raise ValueError(f"{chart_type} requires a group_by column.")
    if group_by and chart_type in {"pie", "scatter", "histogram", "boxplot", "heatmap"}:
        raise ValueError(f"{chart_type} does not support grouped data.")

    filter_column = spec.get("filter_column")
    filter_operator = spec.get("filter_operator")
    filter_value = spec.get("filter_value")
    if filter_column:
        legacy_filter = _validate_text_filter(
            df,
            {
                "column": filter_column,
                "operator": filter_operator,
                "value": filter_value,
            },
        )
        spec["filter_operator"] = legacy_filter["operator"]
        spec["filter_value"] = legacy_filter["value"]
    elif filter_operator is not None or filter_value is not None:
        raise ValueError("Text filter settings require a filter_column.")

    text_filters = spec.get("text_filters") or []
    if not isinstance(text_filters, list):
        raise TypeError("text_filters must be a list.")
    if len(text_filters) > MAX_FILTER_RULES:
        raise ValueError(f"A chart can use at most {MAX_FILTER_RULES} text filters.")
    spec["text_filters"] = [
        _validate_text_filter(df, rule) for rule in text_filters
    ]

    exclusions = spec.get("exclude_values") or {}
    if not isinstance(exclusions, dict):
        raise TypeError("exclude_values must be an object keyed by column name.")
    exclusion_count = 0
    for column, values in exclusions.items():
        _require_column(df, column, "exclusion")
        exclusion_count += len(values) if isinstance(values, list) else 1
    if exclusion_count > 100:
        raise ValueError("A chart can exclude at most 100 category values.")

    numeric_filters = spec.get("numeric_filters") or []
    if not isinstance(numeric_filters, list):
        raise TypeError("numeric_filters must be a list.")
    if len(numeric_filters) > MAX_FILTER_RULES:
        raise ValueError(f"A chart can use at most {MAX_FILTER_RULES} numeric filters.")
    for rule in numeric_filters:
        if not isinstance(rule, dict):
            raise TypeError("Each numeric filter must be an object.")
        column = _require_column(df, rule.get("column"), "numeric filter")
        _require_numeric_data(df, column, "numeric filter")
        operator = rule.get("operator")
        if operator not in ALLOWED_NUMERIC_FILTERS:
            raise ValueError("The numeric filter operator is not supported.")
        value = rule.get("value")
        if operator == "between":
            if not isinstance(value, list) or len(value) != 2:
                raise ValueError("A between filter requires exactly two values.")
            converted = pd.to_numeric(pd.Series(value), errors="coerce")
            if converted.isna().any():
                raise ValueError("Numeric filter values must be numbers.")
        else:
            if isinstance(value, (dict, list)):
                raise ValueError("Numeric filter values must be single numbers.")
            if pd.isna(pd.to_numeric(value, errors="coerce")):
                raise ValueError("Numeric filter values must be numbers.")

    sort_order = str(spec.get("sort_order", "auto")).strip().lower()
    if sort_order not in ALLOWED_SORT_ORDERS:
        supported = ", ".join(sorted(ALLOWED_SORT_ORDERS))
        raise ValueError(f"Sort order must be one of: {supported}.")
    spec["sort_order"] = sort_order

    filtered_df = apply_filter(df, spec)
    filtered_df = apply_exclusions(filtered_df, spec)
    filtered_df = apply_numeric_filters(filtered_df, spec)
    if filtered_df.empty:
        raise ValueError("The requested filters leave no rows to chart.")

    if chart_type == "histogram":
        numeric_column = y or x
        _require_numeric_data(filtered_df, numeric_column, "histogram")
    elif chart_type == "scatter":
        numeric_pairs = filtered_df[[x, y]].apply(
            pd.to_numeric,
            errors="coerce",
        ).dropna()
        if numeric_pairs.empty:
            raise ValueError("The requested filters leave no numeric pairs to chart.")
    elif chart_type == "boxplot":
        boxplot_rows = filtered_df[[x, y]].copy()
        boxplot_rows[y] = pd.to_numeric(boxplot_rows[y], errors="coerce")
        if boxplot_rows.dropna().empty:
            raise ValueError("The requested filters leave no boxplot values to chart.")
    elif chart_type != "heatmap" and aggregation != "count":
        _require_numeric_data(filtered_df, y, "y-axis")

    if x and chart_type not in {"scatter", "histogram", "heatmap"}:
        category_count = int(filtered_df[x].nunique(dropna=True))
        category_limit = 12 if chart_type == "pie" else 50
        if category_count > category_limit:
            raise ValueError(
                f"This {chart_type} chart has {category_count} x-axis categories; "
                f"the safe display limit is {category_limit}. Add a filter or choose "
                "a lower-cardinality column."
            )
    if group_by:
        group_count = int(filtered_df[group_by].nunique(dropna=True))
        if group_count > 12:
            raise ValueError(
                f"The grouping column has {group_count} values; the display limit is 12."
            )

    if chart_type == "pie":
        pie_values = aggregate_data(filtered_df, spec)["value"]
        if (pie_values < 0).any() or pie_values.sum() <= 0:
            raise ValueError("Pie chart values must be non-negative and total above zero.")

    return spec


def aggregate_data(df: pd.DataFrame, chart_spec: dict) -> pd.DataFrame:
    """Aggregate data based on the chart specification."""

    x = chart_spec.get("x")
    y = chart_spec.get("y")
    group_by = chart_spec.get("group_by")
    aggregation = chart_spec.get("aggregation", "sum")

    allowed_aggregations = ["sum", "mean", "count"]

    if aggregation not in allowed_aggregations:
        aggregation = "sum"

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
        df = df.copy()
        df[y] = pd.to_numeric(df[y], errors="coerce")
        result = df.groupby(group_columns)[y].agg(aggregation).reset_index()
        result = result.rename(columns={y: "value"})

    sort_order = chart_spec.get("sort_order", "auto")
    if sort_order in {"ascending", "descending"}:
        result = result.sort_values(
            group_columns,
            ascending=sort_order == "ascending",
            kind="stable",
        )
    elif sort_order == "chronological":
        chronological_values = pd.to_datetime(result[x], errors="coerce")
        if chronological_values.notna().all():
            result = (
                result.assign(_chronological_sort=chronological_values)
                .sort_values(["_chronological_sort", *group_columns[1:]], kind="stable")
                .drop(columns="_chronological_sort")
            )

    return result.reset_index(drop=True)


def render_ai_chart(df: pd.DataFrame, chart_spec: dict):
    """Render a chart based on a safe AI-generated chart specification."""

    chart_spec = validate_chart_spec(df, chart_spec)

    chart_type = chart_spec.get("chart_type")
    x = chart_spec.get("x")
    y = chart_spec.get("y")
    group_by = chart_spec.get("group_by")
    title = chart_spec.get("title", "AI Generated Chart")

    filtered_df = apply_filter(df, chart_spec)
    filtered_df = apply_exclusions(filtered_df, chart_spec)
    filtered_df = apply_numeric_filters(filtered_df, chart_spec)

    fig, ax = plt.subplots(figsize=(9, 5))

    if chart_type == "histogram":
        if y and y in filtered_df.columns:
            numeric_col = y
        elif x and x in filtered_df.columns:
            numeric_col = x
        else:
            raise ValueError("Histogram requires a numeric column.")

        values = pd.to_numeric(filtered_df[numeric_col], errors="coerce").dropna()

        ax.hist(values, bins=20)
        ax.set_title(title)
        ax.set_xlabel(numeric_col)
        ax.set_ylabel("Frequency")
        plt.tight_layout()
        return fig

    if chart_type == "boxplot":
        if not x or x not in filtered_df.columns:
            raise ValueError("Boxplot requires a valid categorical x column.")

        if not y or y not in filtered_df.columns:
            raise ValueError("Boxplot requires a valid numeric y column.")

        plot_df = filtered_df[[x, y]].copy()
        plot_df[y] = pd.to_numeric(plot_df[y], errors="coerce")
        plot_df = plot_df.dropna()

        grouped = plot_df.groupby(x)[y]
        groups = [group.values for _, group in grouped]
        labels = [str(label) for label in grouped.groups]

        ax.boxplot(groups, labels=labels)
        ax.set_title(title)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        return fig

    if chart_type == "heatmap":
        numeric_df = filtered_df.select_dtypes(include="number")

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

    if chart_type == "scatter":
        if not x or x not in filtered_df.columns:
            raise ValueError("Scatter plot requires a valid numeric x column.")
        if not y or y not in filtered_df.columns:
            raise ValueError("Scatter plot requires a valid numeric y column.")

        plot_df = filtered_df[[x, y]].copy()
        plot_df[x] = pd.to_numeric(plot_df[x], errors="coerce")
        plot_df[y] = pd.to_numeric(plot_df[y], errors="coerce")
        plot_df = plot_df.dropna()
        if plot_df.empty:
            raise ValueError("Scatter plot has no numeric values to display.")

        ax.scatter(plot_df[x], plot_df[y], alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.grid(alpha=0.25)
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
        plt.tight_layout()
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
            ax.fill_between(
                chart_data[x].astype(str),
                chart_data["value"]
            )

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
            else:
                raise ValueError(f"Unsupported chart type: {chart_type}")

        ax.legend()

    else:
        if chart_type == "line":
            ax.plot(chart_data[x].astype(str), chart_data["value"], marker="o")
        elif chart_type == "bar":
            ax.bar(chart_data[x].astype(str), chart_data["value"])
        else:
            raise ValueError(f"Unsupported chart type: {chart_type}")

    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel("Value")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    return fig
