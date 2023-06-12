from typing import Dict, List

import dash
from dash import Input, Output, callback, dcc, html
from dash.dash_table import DataTable

from jumanji_benchmarks.data import (
    BATCH_SIZES,
    DEFAULT_BATCH_SIZE,
    DEFAULT_PLATFORM,
    PLATFORMS,
)
from jumanji_benchmarks.visuals import table_data

dash.register_page(__name__, path="/", title="Jumanji Benchmarks")

batch_size_input = dcc.Dropdown(BATCH_SIZES, DEFAULT_BATCH_SIZE, className="dropdown")
platform_input = dcc.RadioItems(PLATFORMS, DEFAULT_PLATFORM)
table = DataTable(
    None,
    [
        {"name": "Commit", "id": "commit"},
        {"name": "Steps/Sec", "id": "rate"},
        {"name": "Iterative Improvement", "id": "iterative"},
        {"name": "Total Improvement", "id": "total"},
    ],
    style_cell_conditional=[{"if": {"column_id": "commit"}, "textAlign": "left"}],
)

layout = html.Div(
    [
        html.Div([html.Label("Batch Size:"), batch_size_input]),
        html.Div([html.Label("Platform:"), platform_input]),
        html.Div(table, className="table"),
    ],
    className="content",
)


@callback(
    Output(table, "data"),
    Input(batch_size_input, "value"),
    Input(platform_input, "value"),
)
def update_platform_graphs(batch_size: int, platform: str) -> List[Dict[str, str]]:
    return table_data(batch_size, platform)
