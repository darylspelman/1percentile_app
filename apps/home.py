from dash import html
import dash_bootstrap_components as dbc
# from dash_bootstrap_components._components.NavbarBrand import NavbarBrand


# needed only if running this as a single page app
# =============================================================================
# external_stylesheets = [dbc.themes.LUX]
# app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
# =============================================================================


style_image = {"border":"0px solid black", "margin-bottom": "0px"} # 'display': 'inline-block'}
style_htmlA = {'text-align':'left','vertical-align':'center',"border":"0px solid black"} #, 'overflow':'auto'} #, 'white-space':'nowrap'}
style_col = {"border":"0px solid black"}
style_line = {"border-bottom":"1px solid gray"}
style_text = {
    'text-align': 'left',
    'vertical-align': 'center',
    'border': '0px solid black',
    'white-space': 'nowrap',  # Prevent text wrapping by default
    'overflow': 'hidden',
    'text-overflow': 'ellipsis',  # Show ellipsis (...) for overflowed text
    'max-width': '100%',  # Ensure text doesn't exceed its container
}



# change to app.layout if running as single page app instead
layout = html.Div(
    [
        dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            html.H1("Finding the Best Companies Globally", className="text-center"),
                            className="mb-4 mt-4",
                        )
                    ]
                ),
                dbc.Row(style={"height": "20px"}),
                dbc.Row(
                    [
                        dbc.Col(
                            html.A(
                                [
                                    html.Img(src="/assets/L1.png", height="45px", style=style_image),
                                    dbc.NavbarBrand(["Company Quality"], className="ml-2 navbar-brand-text"),
                                ],
                                style=style_htmlA,
                                href="/quality_company",
                            ),
                            className="mb-3",
                        ),
                        dbc.Col(
                            html.A(
                                [
                                    html.Img(src="/assets/rv-screen.png", height="45px", style=style_image),
                                    dbc.NavbarBrand(["Company Screen"], className="ml-2 navbar-brand-text"),
                                ],
                                style=style_htmlA,
                                href="/quality_screen",
                            ),
                            className="mb-3",
                        ),
                        dbc.Col(
                            html.A(
                                [
                                    html.Img(src="/assets/rv-screen.png", height="45px", style=style_image),
                                    dbc.NavbarBrand(["Metric Screen"], className="ml-2 navbar-brand-text"),
                                ],
                                style=style_htmlA,
                                href="/metric_screen",
                            ),
                            className="mb-3",
                        ),
                        dbc.Col(
                            html.A(
                                [
                                    html.Img(src="/assets/comps.png", height="45px", style=style_image),
                                    dbc.NavbarBrand(["Comparable Companies"], className="ml-2 navbar-brand-text"),
                                ],
                                style=style_htmlA,
                                href="/comp_metrics",
                            ),
                            className="mb-3",
                        ),
                        dbc.Col(
                            html.A(
                                [
                                    html.Img(src="/assets/rv-tear.png", height="45px", style=style_image),
                                    dbc.NavbarBrand(["Fundamentals"], className="ml-2 navbar-brand-text"),
                                ],
                                style=style_htmlA,
                                href="/fundamentals",
                            ),
                            className="mb-3",
                        ),
                    ],style={"margin-left":"10%"}
                ),
                               
                ### Closing padding
                dbc.Row(style={"height": "20px"}),
            ],
            fluid=True,  # Make the container fluid to allow it to resize with the screen
            style={"padding": "20px", "height": "100vh"},  # Add padding around the container
        ),     
    ],
    style={
        "background-image": 'url("assets/airplane_dash_opaque2_3.jpg")',
        "background-repeat": "repeat",
        "background-position": "center",
        "opacity": "1.0",
        "background-size": "cover",
        "height": "100%",
    },
)






# needed only if running this as a single page app
# =============================================================================
# if __name__ == '__main__':
#     app.run_server(debug=True)
# =============================================================================
