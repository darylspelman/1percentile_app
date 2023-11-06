from dash import html
import dash_bootstrap_components as dbc

# needed only if running this as a single page app
# =============================================================================
# external_stylesheets = [dbc.themes.LUX]
# app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
# =============================================================================


style_image = {"border":"0px solid black"} # 'display': 'inline-block'}
style_text = {'text-align':'left','vertical-align':'center',"border":"0px solid black"} #, 'display': 'inline-block'}
style_htmlA = {'text-align':'left','vertical-align':'center',"border":"0px solid black"} #, 'overflow':'auto'} #, 'white-space':'nowrap'}
style_col = {"border":"0px solid black"}
style_line = {"border-bottom":"1px solid gray"}



# change to app.layout if running as single page app instead
layout = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col(html.H1("Finding the Best Companies Globally", className="text-center")
                    , className="mb-5 mt-5", style={'padding-top':'20px'})
        ]),
        
### COMPANY FUNDAMENTALS
        dbc.Row(style={'height':'60px'}),  

        dbc.Row([
            dbc.Col(html.A([
                    html.Img(src="/assets/L1.png", height="45px", style=style_image),
                    dbc.NavbarBrand(["Company Quality"], className="ml-2", style=style_text),
                        ], style=style_htmlA, href="/quality_company"),width=3, style=style_col),            
            dbc.Col(html.A([
                    html.Img(src="/assets/rv-screen.png", height="45px", style=style_image),
                    dbc.NavbarBrand(["Company Screen"], className="ml-2", style=style_text),
                        ], style=style_htmlA, href="/quality_screen"),width=3, style=style_col)          
        ]),


### Closing padding        
        dbc.Row(style={'height':'200px'})
        
        ], style={"height": "100vh"})

        ], 
        style={
       # 'background-image': 'url("assets/airplane_dash_opaque2_3.jpg")',
        'background-repeat': 'repeat',
        'background-position': 'center',
        'opacity':'1.0',
        'background-size':'cover',
        'height':'100%'
        }
        )






# needed only if running this as a single page app
# =============================================================================
# if __name__ == '__main__':
#     app.run_server(debug=True)
# =============================================================================
