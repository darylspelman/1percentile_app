# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 20:01:03 2023

@author: daryl
"""

from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate

import pandas as pd

#Needed when running in full environment
from app import app




# Load the datafile
df = pd.read_pickle('data/business_quality_data.pkl')
df.reset_index(inplace=True)

# Create a smaller dataframe with only required columnns and formatted
df = df[['index', 'name', 'mar_cap_mUSD', '3m_val_turnover_mUSD']]
df['combined'] = df['index'].astype(str) + df['name'].astype(str)
df['mar_cap_mUSD'] = df['mar_cap_mUSD']/1000
df['mar_cap_mUSD'] = df['mar_cap_mUSD'].apply(lambda x: f"{x:,.1f}" if pd.notna(x) else x)
df['3m_val_turnover_mUSD'] = df['3m_val_turnover_mUSD'].apply(lambda x: f"{x:,.1f}" if pd.notna(x) else x)


# The APP
# needed only if running this as a single page app
# =============================================================================
# app = dash.Dash(__name__)
# =============================================================================


## DEFINE THE CHART LAYOUT
layout = html.Div([
     dbc.Container([
         dbc.Row(
             dbc.Col(html.H1("Ticker Lookup"
                     , className="text-center")
                     , className="mb-5 mt-5", style={'padding-top':'20px'})
         ),
        
        # Get the ticker input
        html.Div([
            dcc.Input(id='ticker-input-lookup', value='', type='text', style={'margin-right': '10px'}, placeholder='Type here...')
        ], style={'padding': '10px'}),

        # Give ticker name and some text

    
        # SUMMARY TABLE

        html.Div([
        dash_table.DataTable(id='ticker_lookup',
                             style_table={'overflowY': 'auto'},
                             style_header = {
                                    'backgroundColor': '#3E5C7B',
                                    'color': 'white',
                                    'fontWeight': 'bold',
                                    'whiteSpace': 'normal',  # Enable text wrapping,
                                    'textAlign': 'left'
                                },
                            style_cell={
                                    'textAlign': 'left',  # Align text to the left in the cells
                                    'whiteSpace': 'normal',
                                    'overflow': 'hidden',
                                    'textOverflow': 'ellipsis',
                                    'maxWidth': 0  # Adjust as needed, 0 means the width is set by the content up to the table's max width
                                },
                            style_cell_conditional=[
                                    # Default width for all columns
                                    {'if': {'column_id': 'index'},
                                     'minWidth': '100px', 'width': '20%', 'maxWidth': '100px'},
                                    
                                    # Double width for the second column
                                    {'if': {'column_id': 'name'},
                                     'minWidth': '200px', 'width': '40%', 'maxWidth': '200px'},
                                
                                    # Default width for other columns
                                    {'if': {'column_id': 'mar_cap_mUSD'},
                                     'minWidth': '100px', 'width': '20%', 'maxWidth': '100px'},
                                    {'if': {'column_id': '3m_val_turnover_mUSD'},
                                     'minWidth': '100px', 'width': '20%', 'maxWidth': '100px'},
                                ],
                            columns=[
                                {'name': 'Ticker', 'id': 'index'},
                                {'name': 'Name', 'id': 'name'},
                                {'name': 'Market Cap, b USD', 'id': 'mar_cap_mUSD'},
                                {'name': 'ADV, m USD', 'id': '3m_val_turnover_mUSD'}]
                             ),
                ]),

    ])
])


@app.callback(
    Output('ticker_lookup', 'data'),
    [Input('ticker-input-lookup', 'value')]
)
def update_table(search_value):
    if search_value is None or search_value == "":
        # Prevent update if the input field is empty
        raise PreventUpdate

    # Filter the DataFrame for partial matches
    filtered_df = df[df['combined'].str.contains(search_value, case=False, na=False)]

    # Limit the results to 20 entries
    filtered_df = filtered_df.head(20)

    filtered_df = filtered_df[['index', 'name', 'mar_cap_mUSD', '3m_val_turnover_mUSD']]
    # Convert the DataFrame to a format suitable for the DataTable
    data = filtered_df.to_dict('records')

    return data






# needed only if running this as a single page app
# =============================================================================
# if __name__ == '__main__':
#     app.run_server(debug=True)
# =============================================================================

