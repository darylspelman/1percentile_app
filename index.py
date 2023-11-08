from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

# must add this line in order for the app to be deployed successfully on Heroku
from app import server
from app import app

# import all pages in the app
from apps import home, quality_company, quality_screen, metric_screen


#from memory_profiler import profile




# Define the dropdown menu on all the pages
# https://github.com/facultyai/dash-bootstrap-components/blob/master/examples/advanced-component-usage/Navbars.py
dropdown = dbc.DropdownMenu(
    children=[
        dbc.DropdownMenuItem("Home", href="/home"),
        dbc.DropdownMenuItem(divider=True),

        dbc.DropdownMenuItem("Company Quality", href="/quality_company"),
        dbc.DropdownMenuItem("Company Screen", href="/quality_screen"),
        dbc.DropdownMenuItem("Metric Screen", href="/metric_screen"),
    ],
    nav = True,
    in_navbar = True,
    label = "Menu",
)

# Setup the navigation bar at the top of the page
navbar = dbc.Navbar(
    dbc.Container(
        [
            html.A(
                # Use row and col to control vertical alignment of logo / brand
                dbc.Row(
                    [
                        dbc.Col(html.Img(src="/assets/volatility_white.png", height="45px")),
                        dbc.Col(dbc.NavbarBrand("One Percentile", className="ml-2", style={'text-decoration': 'none'})),
                    ],
                    align="center",
  #                  no_gutters=True,
                ),
                href="/home", style={'text-decoration': 'none'}
            ),
            dbc.NavbarToggler(id="navbar-toggler2"),
            dbc.Collapse(
                dbc.Nav(
                    # right align dropdown menu with ml-auto className
                    [dropdown], className="ml-auto", navbar=True
                ),
                id="navbar-collapse2",
                navbar=True,
            ),
        ]
    ),
    color="dark",
    dark=True,
    className="",
)

#@profile
def toggle_navbar_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

for i in [2]:
    app.callback(
        Output(f"navbar-collapse{i}", "is_open"),
        [Input(f"navbar-toggler{i}", "n_clicks")],
        [State(f"navbar-collapse{i}", "is_open")],
    )(toggle_navbar_collapse)

# embedding the navigation bar
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    navbar,
    html.Div(id='page-content')
])


@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):   
    if pathname == '/quality_company':
        return quality_company.layout
    elif pathname == '/quality_screen':
        return quality_screen.layout
    elif pathname == '/metric_screen':
        return metric_screen.layout
    else:
        return home.layout

if __name__ == '__main__':
    app.run_server(debug=True)