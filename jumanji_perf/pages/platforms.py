import dash
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html

from jumanji_perf.visuals import (
    COMMITS,
    DEFAULT_COMMIT,
    DEFAULT_WRAPPER,
    WRAPPERS,
    figures_by_platform,
)

dash.register_page(__name__)

commit_input = dcc.Dropdown(COMMITS, DEFAULT_COMMIT, className="commit-dropdown")
wrapper_input = dcc.RadioItems(WRAPPERS, DEFAULT_WRAPPER)
graph = dcc.Graph(figure=go.Figure(), className="run-graph")

layout = html.Div(
    [
        html.Div([html.Label("Commit:"), commit_input]),
        html.Div([html.Label("Wrapper:"), wrapper_input]),
        graph,
    ],
    className="content",
)


@callback(
    Output(graph, "figure"),
    Input(commit_input, "value"),
    Input(wrapper_input, "value"),
)
def update_platform_graphs(commit_hash: str, wrapper: str) -> go.Figure:
    return figures_by_platform(commit_hash, wrapper)
