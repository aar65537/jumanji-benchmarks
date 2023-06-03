import dash
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html

from jumanji_perf.visuals import COMMITS, PLATFORMS, figures_by_wrappers

dash.register_page(__name__)

commit_input = dcc.Dropdown(COMMITS, COMMITS[0])
platform_input = dcc.RadioItems(PLATFORMS, PLATFORMS[0])
run_graph = dcc.Graph(figure=go.Figure())
compile_graph = dcc.Graph(figure=go.Figure())

layout = html.Div([commit_input, platform_input, run_graph, compile_graph])


@callback(
    Output(run_graph, "figure"),
    Output(compile_graph, "figure"),
    Input(commit_input, "value"),
    Input(platform_input, "value"),
)
def update_wrapper_graphs(commit_hash: str, platform: str) -> go.Figure:
    return figures_by_wrappers(commit_hash, platform)
