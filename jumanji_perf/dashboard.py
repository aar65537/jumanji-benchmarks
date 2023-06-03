import dash
from dash import dcc, html

dashboard = dash.Dash(__name__, title="Jumanji Benchmarks", use_pages=True)

dashboard.layout = html.Div(
    [
        html.Div(
            [
                html.H1("Jumanji Performance Benchmarks"),
                html.H2("compare by"),
                html.Div(
                    [
                        html.Div(
                            dcc.Link(f"{page['name']}", href=page["relative_path"])
                        )
                        for page in list(dash.page_registry.values())[1:]
                    ]
                ),
            ]
        ),
        dash.page_container,
    ]
)


if __name__ == "__main__":
    dashboard.run_server(debug=True)
