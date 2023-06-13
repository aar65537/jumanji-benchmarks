import dash
from dash import dcc, html

dashboard = dash.Dash(__name__, title="Jumanji Benchmarks", use_pages=True)

header = html.H1(dcc.Link("Jumanji Benchmarks", href="/"))
subheader = html.H2("compare by")
pages = html.Div(
    [
        dcc.Link(f"{page['name']}", href=page["relative_path"])
        for page in list(dash.page_registry.values())[1:]
    ],
    className="links",
)

sidebar = html.Div([header, subheader, pages], className="sidebar")

dashboard.layout = html.Div([sidebar, dash.page_container])


if __name__ == "__main__":
    dashboard.run_server(host="0.0.0.0", debug=True)
