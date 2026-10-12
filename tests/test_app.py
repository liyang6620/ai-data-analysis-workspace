from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_app_opens_with_demo_dataset():
    app_path = Path(__file__).parents[1] / "app.py"
    app = AppTest.from_file(app_path)

    app.run(timeout=15)

    assert not app.exception
    assert app.header[0].value == "Dataset Overview"
    assert [(metric.label, metric.value) for metric in app.metric] == [
        ("Rows", "12"),
        ("Columns", "6"),
        ("Numeric Columns", "3"),
        ("Categorical Columns", "3"),
    ]
