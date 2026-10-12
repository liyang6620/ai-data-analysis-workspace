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


def test_ai_workspace_preserves_original_controls():
    app_path = Path(__file__).parents[1] / "app.py"
    app = AppTest.from_file(app_path)

    app.run(timeout=15)
    app.radio[1].set_value("AI Workspace").run(timeout=15)

    assert not app.exception
    assert app.session_state["ai_tool"] == "assistant"
    labels = [button.label for button in app.button]
    assert "Ask AI Assistant" in labels
    assert "Generate AI Chart" in labels
