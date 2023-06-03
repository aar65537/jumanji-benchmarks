import dash
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html

from jumanji_perf.visuals import COMMITS, WRAPPERS, figures_by_platform

dash.register_page(__name__)

run_graph = dcc.Graph(figure=go.Figure())
compile_graph = dcc.Graph(figure=go.Figure())
commit_input = dcc.Dropdown(COMMITS, COMMITS[0])
wrapper_input = dcc.RadioItems(WRAPPERS, WRAPPERS[0])

layout = html.Div([commit_input, wrapper_input, run_graph, compile_graph])


@callback(
    Output(run_graph, "figure"),
    Output(compile_graph, "figure"),
    Input(commit_input, "value"),
    Input(wrapper_input, "value"),
)
def update_platform_graphs(commit_hash: str, wrapper: str) -> go.Figure:
    return figures_by_platform(commit_hash, wrapper)
