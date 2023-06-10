import dash
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html

from jumanji_perf.visuals import (
    DEFAULT_PLATFORM,
    DEFAULT_WRAPPER,
    PLATFORMS,
    WRAPPERS,
    figures_by_commit,
)

dash.register_page(__name__)

platform_input = dcc.RadioItems(PLATFORMS, DEFAULT_PLATFORM)
wrapper_input = dcc.RadioItems(WRAPPERS, DEFAULT_WRAPPER)
graph = dcc.Graph(figure=go.Figure(), className="run-graph")

layout = html.Div(
    [
        html.Div([html.Label("Platform:"), platform_input]),
        html.Div([html.Label("Wrapper:"), wrapper_input]),
        graph,
    ],
    className="content",
)


@callback(
    Output(graph, "figure"),
    Input(platform_input, "value"),
    Input(wrapper_input, "value"),
)
def update_platform_graphs(platform: str, wrapper: str) -> go.Figure:
    return figures_by_commit(platform, wrapper)
