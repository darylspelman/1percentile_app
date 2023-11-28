# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 20:01:03 2023

@author: daryl
"""

import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

import numpy as np
import pandas as pd

from scipy.stats import percentileofscore

#Needed when running in full environment
from app import app



def create_IS_table(ticker):
    """
    Creates the custom table for the income statement
    """
    # Get the data for the specific ticker
    df_out = df.xs(ticker, level=1).transpose()
    
    
    # METRIC CALCULATION
    df_out.loc['Revenue Growth'] = (df_out.loc['Total Revenue'].pct_change()*100)
    df_out.loc['Gross Margin'] = (df_out.loc['Gross Profit'] / df_out.loc['Total Revenue'] *100)
    df_out.loc['OPM'] = (df_out.loc['Operating Income'] / df_out.loc['Total Revenue'] *100)
    df_out.loc['OP Growth'] = (df_out.loc['Operating Income'].pct_change()*100)
    df_out.loc['Profit Margin'] = (df_out.loc['Net Income'] / df_out.loc['Total Revenue'] *100)
    
    df_out.loc['COGS'] = df_out.loc['Total Revenue'] - df_out.loc['Gross Profit']
    df_out.loc['Tax'] = df_out.loc['Pretax Income'] - df_out.loc['Net Income']
    df_out.loc['ETR'] = (df_out.loc['Tax'] / df_out.loc['Pretax Income'] *100)  
    
    df_out.loc['R&D % sales'] = (df_out.loc['Research & Development'] / df_out.loc['Total Revenue'] *100)
    df_out.loc['S&M % sales'] = (df_out.loc['Selling & Marketing Expense'] / df_out.loc['Total Revenue'] *100)
    df_out.loc['Total SG&A % sales'] = (df_out.loc['Selling General and Administrative'] / df_out.loc['Total Revenue'] *100)
     
    df_out.loc['EPS'] = df_out.loc['Net Income'] / df_out.loc['Diluted Average Shares']
    df_out.loc['EPS Growth'] = (df_out.loc['Operating Income'].pct_change()*100)
    df_out.loc['Share Growth'] = (df_out.loc['Diluted Average Shares'].pct_change()*100)
    
    # FINAL FORMATTING
    df_out.loc['SMALL1'] = np.nan
    df_out.loc['SMALL2'] = np.nan
    df_out.loc['SMALL3'] = np.nan
    df_out.loc['SMALL4'] = np.nan
    df_out.loc['SMALL5'] = np.nan
    
    # Table order
    new_order = ['Total Revenue', 'Revenue Growth', 'SMALL1', 
                 'COGS', 'Gross Profit', 'Gross Margin', 'SMALL2',
                 'Research & Development', 'R&D % sales','Selling & Marketing Expense', 'S&M % sales', 'Selling General and Administrative', 'Total SG&A % sales', 
                 'Operating Income', 'OPM', 'OP Growth', 'SMALL3', 
                 'Interest Income','Interest Expense','SMALL4', 
                 'Pretax Income', 'Tax', 'ETR', 'Net Income', 'Profit Margin', 'Net Income Common Stockholders', 'SMALL5', 
                 'Diluted Average Shares', 'Share Growth', 'EPS', 'EPS Growth']
    df_final = df_out.reindex(new_order)
    
    # List of rows to format with thousand separators
    row_list = ['Total Revenue', 'COGS', 'Gross Profit', 'Operating Income', 'Research & Development', 'Selling & Marketing Expense', 
                'Selling General and Administrative', 'Pretax Income', 'Net Income', 'Diluted Average Shares', 'Tax', 'Net Income Common Stockholders']
    
    for row in row_list:
        df_final.loc[row] = df_final.loc[row].apply(lambda x: "{:,.0f}".format(x) if isinstance(x, (int, float)) and not pd.isna(x) else x)

    # Create a new list with items from new_order that are not in row_list. Format these to 1 decimal place
    decimal_place = [item for item in new_order if item not in row_list]
    decimal_place.remove('EPS') # Remove the EPS row for separate formatting
    for row in decimal_place:
        df_final.loc[row] = df_final.loc[row].apply(lambda x: "{:.1f}".format(x) if pd.notna(x) else x)

    # Round EPS to 2 decimal places
    df_final.loc['EPS'] = df_final.loc['EPS'].apply(lambda x: "{:.2f}".format(x) if pd.notna(x) else x)
    
    # Rename the columns
    df_final.columns = df_out.loc['period_code']    
    
    
    # CACLCULATE PERCENTILES IN FINAL COLUMN
    df_final['percentile'] = np.nan 
    
    # Dictionary of output rows to calculate percentile
    row_name = {'Revenue Growth':'Revenue 1-yr CAGR', 
                'Gross Margin': 'GPM%',
                'OPM':'OPMargin',
                'R&D % sales':'R&D_pctSales',
                'S&M % sales':'S&M%sales',
                'Profit Margin': 'PM'}
    
    for row in row_name:
        df_final.loc[row,'percentile'] = percentileofscore(df_stocks[row_name[row]].dropna(), df_stocks.loc[ticker, row_name[row]]).round(1) 
    
    
    # RENAME SELECTED INDEX ITEMS WITH SUITABLE TAGS
    new_names = {'Revenue Growth':'Revenue Growth   ',
                 'Total Revenue':'Revenue\t',
                 'Gross Profit':'Gross Profit\t',
                 'Gross Margin':'Gross Margin   ',
                 'R&D % sales':'R&D % sales   ',
                 'S&M % sales':'S&M % sales   ',
                 'Selling General and Administrative': 'Total SG&A Expense',
                 'Total SG&A % sales':'Total SG&A % sales   ',
                 'Operating Income':'Operating Income\t',
                 'OPM':'OPM   ',
                 'OP Growth': 'OP Growth   ',
                 'ETR':'ETR   ',
                 'Profit Margin':'Profit Margin   ',
                 'Net Income':'Net Income\t',
                 'EPS': 'EPS\t',
                 'EPS Growth':'EPS Growth   ',
                 'Share Growth': 'Share Growth   '}
    
    # Updating the index using the dictionary
    df_final = df_final.rename(index=new_names)
    
    
    print('\n', df_final)
    
    # Finally reset index and return 
    df_return = df_final.reset_index()
    df_return.columns = ['Metric'] + list(df_final.columns)
    
    return df_return




# Load the datafile
df_stocks = pd.read_pickle('data/business_quality_data.pkl')
df = pd.read_pickle('data/fundamental_data.pkl')


style_data = {
    'height': '30px',  # Set a smaller default height
}

style_cell = {
    'whiteSpace': 'normal',
    'height': 'auto'
}




# The APP
# needed only if running this as a single page app
# =============================================================================
# app = dash.Dash(__name__)
# =============================================================================


## DEFINE THE CHART LAYOUT
layout = html.Div([
     dbc.Container([
         dbc.Row(
             dbc.Col(html.H1("Quality Company Overview"
                     , className="text-center")
                     , className="mb-5 mt-5", style={'padding-top':'20px'})
         ),
        
        # Get the ticker input and print the name
        # dcc.Input(id='ticker-input', value='AAPL', type='text'),  # Default value is AAPL
        # html.Button(id='submit-button', n_clicks=0, children='Submit'),

        html.Div([
            dcc.Input(id='ticker-input-fund', value='AAPL', type='text', style={'margin-right': '10px'}),  # Default value is AAPL
            html.Button(id='submit-button-fund', n_clicks=0, children='Submit', style={'margin-left': '10px'}),
        ], style={'padding': '10px'}),

        # Give ticker name and some text
        dbc.Row(dbc.Col(html.H2(id='ticker-message-fund'                     
                 , className="text-center")
                , className="mb-3 mt-3")),  # This will display the message style={'padding-top':'20px'}
        dbc.Row(dbc.Col(html.Div(id='stock-info-fund'                     
                 , className="text-left")
                , className="mb-3 mt-3")),  # This will display the message style={'padding-top':'20px'}
        


        # SUMMARY TABLE
        dbc.Row(dbc.Col(html.H2("Income Statement", className="text-left"), 
                         className="mb-3 mt-3", style={'padding-top':'20px'})),
        
        html.Div([
        dash_table.DataTable(id='summ-table-fund',
                             style_table={'overflowY': 'auto'},
                             style_header = {
                                    'backgroundColor': '#3E5C7B',
                                    'color': 'white',
                                    'fontWeight': 'bold',
                                    'whiteSpace': 'normal'  # Enable text wrapping
                                },
                             style_data=style_data,
                             style_cell=style_cell,
                             ),
                ]),

    ])
])



# Second callback for the spider plot and the message
@app.callback(
    [Output('ticker-message-fund', 'children'),
     Output('stock-info-fund', 'children')],  # Two outputs: one for the plot and one for the message
    [Input('submit-button-fund', 'n_clicks')],
    [dash.dependencies.State('ticker-input-fund', 'value')]
)
def update_figure_and_message(n_clicks, ticker):
    ticker = ticker.upper()
    if ticker not in df_stocks.index:
        return 'Ticker not in database', None  # No update to the graph, message updated
    else:
        name = df_stocks.loc[ticker]['name'] + ' (' + ticker + ')'
        info = [
                f"Market Cap (bUSD):  {df_stocks.loc[ticker]['mar_cap_mUSD']/1000:,.1f}", html.Br(),
                f"ADV (mUSD): {df_stocks.loc[ticker]['3m_val_turnover_mUSD']:,.1f}", html.Br(),
                f"Sector: {df_stocks.loc[ticker]['sector']}", html.Br(),
                f"Industry: {df_stocks.loc[ticker]['industry']}", html.Br(), html.Br(),
                html.B(f"All financial numbers in millions {df_stocks.loc[ticker]['reporting_currency'][0]}")
              ]
        return name, info  # Graph updated, message updated with name



@app.callback(
    [Output('summ-table-fund', 'data'),
      Output('summ-table-fund', 'columns'),
      Output('summ-table-fund', 'style_data_conditional')],
    [Input('submit-button-fund', 'n_clicks')],
    [dash.dependencies.State('ticker-input-fund', 'value')]
)
def update_table(n_clicks, ticker):
    ticker = ticker.upper()
    if ticker not in df_stocks.index:
        return [None for _ in range(3)]  # Return an empty figure if no ticker is provided



    # Get data
    data = create_IS_table(ticker)
    
    # # Define columns
    columns = [{'name': col, 'id': col} for col in data.columns]

    # Define the upper and lower thresholds and colours for the percentile columns in the chart
    threshold_upper = 85
    threshold_lower = 15
    colour_upper = '#B0CAC4'
    colour_lower = '#D08452'
    
    style_data_conditional = [
        # Set column relative width of first column. Rest same width
        {'if': {'column_id': 'Metric'},
          'width': '40%',
          'whiteSpace': 'normal',
          'textAlign': 'left',
          },  # Enable text wrapping

         # Bold rows where 'Metric' contains a '\t' instruction
        {
            'if': {
                'filter_query': '{Metric} contains "\t"',
            },
            'fontWeight': 'bold'
        },
    
        # Italicize rows where 'Metric' contains '   '
        {
            'if': {
                'filter_query': '{Metric} contains "   "', # 3 spaces = 
            },
            'fontStyle': 'italic',
            'paddingLeft': '45px'
        },
    
        # Reduce font size for rows where 'Metric' contains 'SMALL'
        {
            'if': {
                'filter_query': '{Metric} contains "SMALL"',
            },
            'color': '#DEE5ED',
            'backgroundColor': '#DEE5ED'
        },

        #Highlight the outliers by row
        {'if': {'column_id': 'percentile',
            'filter_query': f'{{percentile}} > {threshold_upper}'},
          'backgroundColor': colour_upper},       
        {'if': {'column_id': 'percentile',
            'filter_query': f'{{percentile}} < {threshold_lower} && {{percentile}} > 0'},
          'backgroundColor': colour_lower},
    
        
        
    ]    
    # Drop the '000' tag from the column Metric before returning it
    # data['Metric'] = data['Metric'].str.replace('  ', '')
    
    
    return data.to_dict('records'), columns, style_data_conditional



# needed only if running this as a single page app
# =============================================================================
# if __name__ == '__main__':
#     app.run_server(debug=True)
# =============================================================================

