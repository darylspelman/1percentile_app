import dash
import dash_bootstrap_components as dbc
# import dash_auth

# bootstrap theme
# https://bootswatch.com/lux/
external_stylesheets = [dbc.themes.LUX]


# USERNAME_PASSWORD_PAIRS = [['vol','vol']]


app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
# auth = dash_auth.BasicAuth(app, USERNAME_PASSWORD_PAIRS)
app.title = "One Percentile"


server = app.server

app.config.suppress_callback_exceptions = True