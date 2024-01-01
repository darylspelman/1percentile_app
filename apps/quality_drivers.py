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
import plotly.graph_objs as go

from scipy.stats import percentileofscore

#Needed when running in full environment
from app import app


##################
# EDITS TO BE MADE
# Add description field
# Add Summary IS, BS, CFS
# Add S&M %, capex% sales, FCF % sales, CFO % sales
# Fix cash conversion calculation (subtract payables)
# Allow data enty for ticker to be in lower case letters
##################



def generate_histogram(ticker, metric, bins):
    """
    Create the histogram outputs
    """
    
    # Prep the data for the specifc metric
    data = df[metric].replace([np.inf, -np.inf], np.nan).dropna()
    specific_value = df.loc[ticker][metric]
    percentile = percentileofscore(data, specific_value)
    
    # If not a score metric, then trim the ends of the histogram to remove extreme outliers from the plot
    if metric in ['BQ_score','EQ_score','V_score', 'govern_score']:
        custom_text = f'Val: {specific_value:.0f}<br>%ile: {percentile:.0f}'
        height_val=300
    else:
        custom_text = f'Val: {specific_value:.2f}<br>%ile: {percentile:.0f}'
        lower_bound = data.quantile(0.025)
        upper_bound = data.quantile(0.975)
        filtered_series = data[(data >= lower_bound) & (data <= upper_bound)]
        data = filtered_series
        height_val=250

    fig = go.Figure()
    # Compute histogram using numpy first to get the bin counts
    hist, bin_edges = np.histogram(data, bins=bins)
    max_y = np.max(hist)  # Corrected to calculate the maximum value in hist
    
    hist_trace = go.Histogram(x=data, name=metric, marker=dict(color='#558BB8', opacity=1), nbinsx=bins)
    fig.add_trace(hist_trace)

    if np.isnan(specific_value) != True:
        fig.add_shape(go.layout.Shape(type='line', x0=specific_value, x1=specific_value, y0=0, y1=max_y, line=dict(color='red', width=2)))
        fig.add_annotation(text=custom_text, xref="paper", yref="paper", x=0.05, y=0.95, showarrow=False, align="left", bgcolor="white", bordercolor="black", borderwidth=1, font=dict(size=12))
    
    # Lookup chart title
    chart_title = metric_name.loc[metric]['Name']
    
    
    fig.update_layout(title={'text': chart_title, 'y':0.95, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'},
                      height = height_val,
                      margin=dict(l=20, r=20, t=50, b=20),
                      modebar_remove=['zoom', 'pan', 'toImage', 'lasso2d', 'zoomOut2d', 'zoomIn2d', 'resetScale2d', 'autoScale2d', 'select2d']
                      )
    
    # fig.show(config={"displayModeBar": False})
    return fig



def generate_layout_children(graph_ids, per_line, height):
    """
    Generates a row of charts based on the ids, the number of charts and the heights
    """
    width_percent = 100 / per_line  # calculate the width based on how many charts per line
    return [
        html.Div([dcc.Graph(id=graph_id)], style={
            'width': f'{width_percent}%',
            'height': f'{height}px',  # You can adjust this value as needed
            'display': 'inline-block',
            'padding': '0px',
            'margin': '0px'
        })
        for graph_id in graph_ids
    ]




# Load the datafile
df = pd.read_pickle('data/business_quality_data.pkl')
metric_name = pd.read_csv('inputs/metric_name.csv', index_col='Metric')


# Define the score historgrams for the first row
metric_list_A = ['BQ_score', 'V_score', 'EQ_score', 'govern_score']
graphs1=[f'histogram-{i+1}' for i in range(len(metric_list_A))]



# Define the histogram plots for the individual metric plots. Each section has it's own list
#Capital Structure
met_roc = ['ROE', 'median ROE 3-yr', 'ROE_change', 'ROA', 'RNOA', 'ROIC'] #, 'ROCE']#, 'FCF%sales'] #'ROIC']
g_roc = [f'hist_roc-{i+1}' for i in range(len(met_roc))]

#Dupont
met_dupont = ['PM', 'sales/assets', 'assets/equity']
g_dupont = [f'hist_dupont-{i+1}' for i in range(len(met_dupont))]

#Growth
met_growth = ['Revenue 1-yr CAGR', 'Revenue 3-yr CAGR', 'ShareholderEquity 1-yr CAGR', 'ShareholderEquity 3-yr CAGR']
g_growth = [f'hist_growth-{i+1}' for i in range(len(met_growth))]

#Profitability
met_profit = ['GPM%', 'OPMargin', 'OPM_change', 'PM', 'median PM 3-yr', 'PM_change', 'SGA_pctSales', 'FCF%sales', 'CFO%sales']
g_profit = [f'hist_profit-{i+1}' for i in range(len(met_profit))]

#Asset Intensity
met_ai = ['sales/assets', 'sales/asset_change', 'Sales/avg_NOA']
g_ai = [f'hist_ai-{i+1}' for i in range(len(met_ai))]

# Capital Structure
met_cap = ['assets/equity', 'assets/equity_change', 'net_debt/equity']
g_cap = [f'hist_cap-{i+1}' for i in range(len(met_cap))]

# Reinvestment
met_inv = ['R&D_pctSales', 'S&M%sales', 'capitalExpenditures_pctSales', 'total_investment/sales']
g_inv = [f'hist_inv-{i+1}' for i in range(len(met_inv))]

# Shareholder Transaction
met_share = ['Share growth', 'div_payout_rate']
g_share = [f'hist_share-{i+1}' for i in range(len(met_share))]

# Valuation
met_val = ['P/E_ttm', 'P/B', 'P/S_ttm', 'EV/EBIT_ttm', 'EV/EBITDA_ttm']
g_val = [f'hist_val-{i+1}' for i in range(len(met_val))]

# Earnings Quality
met_eq = ['days_CCC', 'days_CCC_change', 'days_sales_outstanding', 'days_inventory_outstanding', 'days_payables_outstanding', 
          'sloan_accruals_assets', 'CFO_NI', 'avg_PPE_life', 'intangibleAssets/totalAssets']
g_eq = [f'hist_eq-{i+1}' for i in range(len(met_eq))]



# Combine all metrics together
met_comb = met_roc + met_dupont + met_growth + met_profit + met_ai + met_cap + met_inv + met_share + met_val + met_eq
g_comb = g_roc + g_dupont + g_growth + g_profit + g_ai + g_cap + g_inv + g_share + g_val + g_eq


# Define number of charts per row for histogram metrics
chart_per_row = 3
height_per_chart = 250

# The APP
# needed only if running this as a single page app
# =============================================================================
# app = dash.Dash(__name__)
# =============================================================================


## DEFINE THE CHART LAYOUT
layout = html.Div([
     dbc.Container([
         dbc.Row(
             dbc.Col(html.H1("Business Quality Drivers"
                     , className="text-center")
                     , className="mb-5 mt-5", style={'padding-top':'20px'})
         ),
        
        # Get the ticker input and print the name
        # dcc.Input(id='ticker-input', value='AAPL', type='text'),  # Default value is AAPL
        # html.Button(id='submit-button', n_clicks=0, children='Submit'),

        html.Div([
            dcc.Input(id='ticker-input-driver', type='text', style={'margin-right': '10px'}),  # Default value is AAPL
            html.Button(id='submit-button-driver', n_clicks=0, children='Submit', style={'margin-left': '10px'}),
        ], style={'padding': '10px'}),

        # Give ticker name and some text
        dbc.Row(dbc.Col(html.H2(id='ticker-message-driver'                     
                 , className="text-center")
                , className="mb-3 mt-3")),  # This will display the message style={'padding-top':'20px'}
        dbc.Row(dbc.Col(html.Div(id='stock-info-driver'                     
                 , className="text-left")
                , className="mb-3 mt-3")),  # This will display the message style={'padding-top':'20px'}
        

        # DRIVER BREAKDOWN
        dbc.Row(dbc.Col(html.H2("Performance Drivers", className="text-left"), 
                         className="mb-3 mt-3", style={'padding-top':'20px'})),
        
        dbc.Row(dbc.Col(html.H3("Return on Capital", className="text-left"), 
                         className="mb-3 mt-3", style={'padding-top':'20px'})),
        *generate_layout_children(g_roc, chart_per_row, height_per_chart),

        dbc.Row(dbc.Col(html.H3("Dupont Decomposition", className="text-left"), 
                         className="mb-3 mt-3", style={'padding-top':'20px'})),
        *generate_layout_children(g_dupont, chart_per_row, height_per_chart),

        dbc.Row(dbc.Col(html.H3("Growth", className="text-left"), 
                         className="mb-3 mt-3", style={'padding-top':'20px'})),        
        *generate_layout_children(g_growth, chart_per_row, height_per_chart),
    
        dbc.Row(dbc.Col(html.H3("Profitability", className="text-left"), 
                         className="mb-3 mt-3", style={'padding-top':'20px'})), 
        *generate_layout_children(g_profit, chart_per_row, height_per_chart),
        
        dbc.Row(dbc.Col(html.H3("Asset Intensity", className="text-left"), 
                         className="mb-3 mt-3", style={'padding-top':'20px'})), 
        *generate_layout_children(g_ai, chart_per_row, height_per_chart),

        dbc.Row(dbc.Col(html.H3("Capital Structure", className="text-left"), 
                         className="mb-3 mt-3", style={'padding-top':'20px'})), 
        *generate_layout_children(g_cap, chart_per_row, height_per_chart),

        dbc.Row(dbc.Col(html.H3("Reinvestment", className="text-left"), 
                         className="mb-3 mt-3", style={'padding-top':'20px'})), 
        *generate_layout_children(g_inv, chart_per_row, height_per_chart),

        dbc.Row(dbc.Col(html.H3("Shareholder Transaction", className="text-left"), 
                         className="mb-3 mt-3", style={'padding-top':'20px'})), 
        *generate_layout_children(g_share, chart_per_row, height_per_chart),
        
        dbc.Row(dbc.Col(html.H3("Valuation", className="text-left"), 
                         className="mb-3 mt-3", style={'padding-top':'20px'})),
        *generate_layout_children(g_val, chart_per_row, height_per_chart),
        
        dbc.Row(dbc.Col(html.H3("Earnings Quality", className="text-left"), 
                         className="mb-3 mt-3", style={'padding-top':'20px'})),
        *generate_layout_children(g_eq, chart_per_row, height_per_chart),
    ])
])


# Set the default value at the start for the Input field
@app.callback(
    Output('ticker-input-driver', 'value'),
    [Input('shared-data', 'data')],
    prevent_initial_call=False  # This is important to prevent the callback from running when the page first loads
)
def set_default_ticker_value(stored_ticker):
    if stored_ticker and 'ticker' in stored_ticker:
        return stored_ticker['ticker']
    return 'MSFT'  # Fallback default value


# Update the dcc.Store value when update button is clicked
@app.callback(
    Output('shared-data', 'data', allow_duplicate=True),
    Input('submit-button-driver', 'n_clicks'),
    dash.dependencies.State('ticker-input-driver', 'value'),
    prevent_initial_call=True
)
def update_shared_data(n_clicks, ticker_value):
    if n_clicks is not None and ticker_value is not None:
        return {'ticker': ticker_value.upper()}
    return dash.no_update


# Third callback for all the histograms
output_args = [Output(graph, 'figure') for graph in g_comb]
@app.callback(
    output_args,
    [Input('submit-button-driver', 'n_clicks'),
     Input('shared-data', 'data')],
    [dash.dependencies.State('ticker-input-driver', 'value')]
)
def update_histograms2(n_clicks, stored_ticker, ticker):
    if ticker == None:
        ticker = stored_ticker['ticker']

    ticker = ticker.upper().strip() 
    if ticker not in df.index:
        return [go.Figure() for _ in range(len(g_comb))]  # Return no updates based on the length of graphs2
    else:
        bins = 100  # Modify as needed
        return [generate_histogram(ticker, metric, bins) for metric in (met_comb)]



# needed only if running this as a single page app
# =============================================================================
# if __name__ == '__main__':
#     app.run_server(debug=True)
# =============================================================================

