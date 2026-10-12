import re

CHART_TERMS = {
    "chart",
    "plot",
    "graph",
    "visualise",
    "visualize",
    "trend",
    "distribution",
    "histogram",
    "scatter",
    "bar chart",
    "line chart",
    "pie chart",
    "boxplot",
    "box plot",
    "heatmap",
    "area chart",
}
CODE_TERMS = {
    "python",
    "pandas",
    "matplotlib",
    "seaborn",
    "ggplot",
    "tidyverse",
    " r code",
    "code",
    "script",
}
CHINESE_CHART_TERMS = {
    "画图",
    "画个图",
    "作图",
    "生成图",
    "生成一个图",
    "图表",
    "可视化",
    "趋势图",
    "分布图",
    "柱状图",
    "折线图",
    "散点图",
    "饼图",
    "热力图",
}
CHINESE_CODE_TERMS = {"代码", "脚本", "用python", "用r"}
UNSUPPORTED_REQUEST_TERMS = {
    "forecast": "forecasting",
    "predict": "forecasting",
    "projection": "forecasting",
    "map": "geospatial maps",
    "geospatial": "geospatial maps",
    "geographic": "geospatial maps",
    "join": "multi-dataset joins",
    "merge datasets": "multi-dataset joins",
    "combine datasets": "multi-dataset joins",
    "dashboard": "custom interactive dashboards",
    "interactive": "custom interactive dashboards",
    "3d": "3D charts",
    "预测": "forecasting",
    "地图": "geospatial maps",
    "合并数据": "multi-dataset joins",
    "仪表盘": "custom interactive dashboards",
    "交互式": "custom interactive dashboards",
}
MAX_CHART_REQUEST_LENGTH = 1000


def _contains_term(text: str, term: str) -> bool:
    if term.isascii():
        return re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text) is not None
    return term in text


def looks_like_chart_request(question: str) -> bool:
    """Route chart intent to the safe renderer unless code is requested explicitly."""

    normalised = re.sub(r"\s+", " ", question.strip().lower())
    if not normalised:
        return False
    if any(
        _contains_term(normalised, term)
        for term in CODE_TERMS | CHINESE_CODE_TERMS
    ):
        return False
    return any(
        _contains_term(normalised, term)
        for term in CHART_TERMS | CHINESE_CHART_TERMS
    )


def validate_chart_request(question: str) -> str:
    """Reject requests outside the renderer's documented product boundary."""

    normalised = re.sub(r"\s+", " ", question.strip().lower())
    if not normalised:
        raise ValueError("Describe the chart you want first.")
    if len(normalised) > MAX_CHART_REQUEST_LENGTH:
        raise ValueError(
            f"Chart requests must be {MAX_CHART_REQUEST_LENGTH} characters or fewer."
        )
    if any(
        _contains_term(normalised, term)
        for term in CODE_TERMS | CHINESE_CODE_TERMS
    ):
        raise ValueError(
            "This tool renders charts and does not generate or execute plotting code. "
            "Use Ask a Question for code guidance."
        )
    for term, capability in UNSUPPORTED_REQUEST_TERMS.items():
        if _contains_term(normalised, term):
            raise ValueError(
                f"This workspace does not support {capability}. Use existing columns "
                "from the current dataset and a supported static chart."
            )
    return normalised
