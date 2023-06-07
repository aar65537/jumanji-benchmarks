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
run_graph = dcc.Graph(figure=go.Figure(), className="run-graph")
compile_graph = dcc.Graph(figure=go.Figure(), className="compile-graph")

layout = html.Div(
    [
        html.Div([html.Label("Platform:"), platform_input]),
        html.Div([html.Label("Wrapper:"), wrapper_input]),
        run_graph,
        compile_graph,
    ],
    className="content",
)


@callback(
    Output(run_graph, "figure"),
    Output(compile_graph, "figure"),
    Input(platform_input, "value"),
    Input(wrapper_input, "value"),
)
def update_platform_graphs(platform: str, wrapper: str) -> go.Figure:
    return figures_by_commit(platform, wrapper)
