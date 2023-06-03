import dash
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html

from jumanji_perf.visuals import PLATFORMS, WRAPPERS, figures_by_commit

dash.register_page(__name__)

run_graph = dcc.Graph(figure=go.Figure())
compile_graph = dcc.Graph(figure=go.Figure())
platform_input = dcc.RadioItems(PLATFORMS, PLATFORMS[0])
wrapper_input = dcc.RadioItems(WRAPPERS, WRAPPERS[0])

layout = html.Div([platform_input, wrapper_input, run_graph, compile_graph])


@callback(
    Output(run_graph, "figure"),
    Output(compile_graph, "figure"),
    Input(platform_input, "value"),
    Input(wrapper_input, "value"),
)
def update_platform_graphs(platform: str, wrapper: str) -> go.Figure:
    return figures_by_commit(platform, wrapper)
