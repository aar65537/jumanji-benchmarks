import dash
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html

from jumanji_perf.visuals import COMMITS, PLATFORMS, figures_by_wrappers

dash.register_page(__name__)

commit_input = dcc.Dropdown(COMMITS, COMMITS[0], className="commit-dropdown")
platform_input = dcc.RadioItems(PLATFORMS, PLATFORMS[0])
run_graph = dcc.Graph(figure=go.Figure(), className="run-graph")
compile_graph = dcc.Graph(figure=go.Figure(), className="compile-graph")

layout = html.Div(
    [
        html.Div([html.Label("Commit:"), commit_input]),
        html.Div([html.Label("Platform:"), platform_input]),
        run_graph,
        compile_graph,
    ],
    className="content",
)


@callback(
    Output(run_graph, "figure"),
    Output(compile_graph, "figure"),
    Input(commit_input, "value"),
    Input(platform_input, "value"),
)
def update_wrapper_graphs(commit_hash: str, platform: str) -> go.Figure:
    return figures_by_wrappers(commit_hash, platform)
