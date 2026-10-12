from io import BytesIO

import pandas as pd
import pytest

from src.agent_controller import (
    generate_auto_insights,
    prepare_dataset_context,
    suggest_local_questions,
)
from src.chart_generator import (
    aggregate_data,
    apply_exclusions,
    apply_filter,
    apply_numeric_filters,
    render_ai_chart,
)
from src.data_loader import DatasetLoadError, load_csv_dataset
from src.quality_checks import run_quality_checks


@pytest.fixture
def sample_frame():
    return pd.DataFrame(
        {
            "region": ["North", "South", "North", "North"],
            "sales": [10, 20, None, 30],
            "constant": ["yes", "yes", "yes", "yes"],
        }
    )


def test_quality_checks_report_missing_and_constant_columns(sample_frame):
    result = run_quality_checks(sample_frame)

    assert result["duplicate_rows"] == 0
    assert result["missing_values"] == {"sales": 1}
    assert result["constant_columns"] == ["constant"]


def test_dataset_context_is_bounded(sample_frame):
    result = prepare_dataset_context(sample_frame)

    assert result["rows"] == 4
    assert result["columns"] == 3
    assert len(result["sample_rows"]) <= 5


def test_text_only_dataset_context_is_supported():
    frame = pd.DataFrame({"category": ["A", "B"], "status": ["open", "closed"]})

    result = prepare_dataset_context(frame)

    assert result["numeric_summary"] == {}
    assert result["top_categories"]["category"] == {"A": 1, "B": 1}


def test_dataset_context_caps_wide_datasets():
    frame = pd.DataFrame({f"column_{index}": [index] for index in range(60)})

    result = prepare_dataset_context(frame)

    assert len(result["column_names"]) == 50
    assert result["columns_omitted"] == 10


def test_filters_and_aggregation(sample_frame):
    filtered = apply_numeric_filters(
        sample_frame,
        {"numeric_filters": [{"column": "sales", "operator": "gte", "value": 20}]},
    )
    filtered = apply_exclusions(filtered, {"exclude_values": {"region": ["South"]}})
    result = aggregate_data(
        filtered,
        {"x": "region", "y": "sales", "aggregation": "sum"},
    )

    assert result.to_dict(orient="records") == [{"region": "North", "value": 30.0}]


def test_text_filter_treats_special_characters_literally():
    frame = pd.DataFrame({"label": ["A.B", "ACB", "A.B extra"]})

    result = apply_filter(
        frame,
        {
            "filter_column": "label",
            "filter_operator": "contains",
            "filter_value": "A.B",
        },
    )

    assert result["label"].tolist() == ["A.B", "A.B extra"]


def test_numeric_filter_accepts_numeric_text(sample_frame):
    result = apply_numeric_filters(
        sample_frame,
        {"numeric_filters": [{"column": "sales", "operator": "gte", "value": "20"}]},
    )

    assert result["sales"].tolist() == [20.0, 30.0]


def test_scatter_chart_preserves_raw_observations():
    frame = pd.DataFrame({"x": [1, 1, 2], "y": [10, 20, 30]})

    figure = render_ai_chart(
        frame,
        {"chart_type": "scatter", "x": "x", "y": "y", "title": "Relationship"},
    )

    offsets = figure.axes[0].collections[0].get_offsets()
    assert offsets.tolist() == [[1.0, 10.0], [1.0, 20.0], [2.0, 30.0]]
    figure.clear()


def test_csv_loader_uses_content_fingerprint():
    first, first_fingerprint = load_csv_dataset(BytesIO(b"a,b\n1,2\n"))
    second, second_fingerprint = load_csv_dataset(BytesIO(b"a,b\n3,4\n"))

    assert first.to_dict(orient="records") == [{"a": 1, "b": 2}]
    assert second.to_dict(orient="records") == [{"a": 3, "b": 4}]
    assert first_fingerprint != second_fingerprint


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (b"", "empty"),
        (b"a,b\n", "no data rows"),
    ],
)
def test_csv_loader_rejects_unusable_files(content, message):
    with pytest.raises(DatasetLoadError, match=message):
        load_csv_dataset(BytesIO(content))


def test_local_questions_do_not_require_ai(sample_frame):
    questions = suggest_local_questions(sample_frame)

    assert len(questions) == 3
    assert any("sales" in question for question in questions)


def test_generic_auto_insights_work_for_any_business_dataset(sample_frame):
    insights = generate_auto_insights(sample_frame)

    assert insights["numeric_summary"]["sales"] == {
        "minimum": 10.0,
        "maximum": 30.0,
        "average": 20.0,
    }
    assert insights["leading_categories"]["region"] == {
        "value": "North",
        "count": 3,
    }
