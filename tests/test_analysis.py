import pandas as pd
import pytest

from src.agent_controller import prepare_dataset_context
from src.chart_generator import aggregate_data, apply_exclusions, apply_numeric_filters
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
    assert len(result["sample_rows"]) <= 10


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
