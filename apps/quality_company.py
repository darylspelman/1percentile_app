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
# Add data table for the metrics
# Add S&M %
##################



def get_spider_data(ticker):
    """
    Takes a ticker and full database and returns a dataframe for plotting the full spider diagram
    """
    df_spider = pd.DataFrame(df.loc[ticker][['BQ_RoC_score_percentile',
            'BQ_growth_score_percentile',
            'BQ_profitability_score_percentile',
            'BQ_asset_intensity_score_percentile',
            'BQ_capital_structure_score_percentile',
            'BQ_reinvestment_score_percentile',
            'BQ_shareholder_score_percentile',

            'V_score_fundamental_percentile',
            'V_score_relative_percentile',

            'EQ_accruals_comp_score_percentile',
            'EQ_assumption_comp_score_percentile',
            'EQ_risk_comp_score_percentile',
            'EQ_compound_comp_score_percentile',

            'Alignement with Shareholders_percentile',
            'Board Quality_percentile',
            'Committee Quality_percentile']])    
    df_spider.columns = ['value']
    df_spider = df_spider * 100
    df_spider['labels'] = ['BQ:\nReturn on Capital',
                            'BQ:\nGrowth',
                            'BQ:\nProfitability',
                            'BQ:\nAsset Intensity',
                            'BQ:\nCapital Structure',
                            'BQ:\nReinvestment',
                            'BQ:\nShareholder Transactions',

                            'Valuation:\nFundamental',
                            'Valuation:\nRelative',
 
                            'EQ:\nAccruals',
                            'EQ:\nAccounting Assumptions',
                            'EQ:\nRisk Indicator',
                            'EQ:\nCompound Metric',

                            'Governance:\nAlignment with Shareholder',
                            'Governance:\nBoard Quality',
                            'Governance:\nCommittee Quality']
    return df_spider



def table_data(ticker):
    """
    Create the summary dataframe for data table
    """
    df_table = pd.DataFrame(df.loc[ticker])
    df_table.columns=['Metric']
    
    # List of metrics for table
    metric_list = ['BQ_score', 'V_score', 'EQ_score', 'govern_score',
                    'ROE', 'median ROE 3-yr', 'ROA', 'RNOA', 'ROCE',
                    'GPM%', 'OPMargin', 'PM', 'median PM 3-yr', 'SGA_pctSales', 'R&D_pctSales',
                    'Revenue 1-yr CAGR', 'ShareholderEquity 1-yr CAGR', 'Revenue 3-yr CAGR', 'ShareholderEquity 3-yr CAGR',
                    'assets/equity', 'net_debt/equity',
                    'sales/assets', 'Sales/avg_NOA',
                    'Share growth', 'div_payout_rate',
                    'days_CCC', 'days_sales_outstanding', 'days_inventory_outstanding', 'days_payables_outstanding',
                    'P/E_ttm', 'P/B', 'P/S_ttm', 'EV/EBIT_ttm', 'EV/EBITDA_ttm']
    
    # Benchmark list and table headings
    peer_group = ['Global', 'All Country', 'Country & Sector', 'Global Industry']
    
    # Cycle through metrics and peer groups
    working = pd.DataFrame(index = metric_list)
    benchmark_list = []
    for peer in peer_group:
        if peer == 'Global':
            benchmark = df
        elif peer == 'All Country':
            benchmark = df[df['country_name'] == df_table.loc['country_name'][0]]
        elif peer == 'Global Industry':
            benchmark = df[df['industry'] == df_table.loc['industry'][0]]
        elif peer == 'Country & Sector':
            benchmark = df[(df['country_name'] == df_table.loc['country_name'][0]) & 
                                  (df['sector'] == df_table.loc['sector'][0])]
        benchmark_list.append(len(benchmark))
    
        percent_list = []
        value_list = []
        for metric in metric_list:
            value = df_table.loc[metric].iloc[0]
            if np.isnan(float(value)):
                percentile = np.nan
            else:
                percentile = percentileofscore(benchmark[metric].dropna(), value)
            
            percent_list.append(round(float(percentile),0))
            value_list.append(value)
        working[peer] = percent_list
    
    # Add number of observations
    working['Value'] = value_list
    toprow = pd.DataFrame([benchmark_list], columns=peer_group, index=['Number companies'])
    working = pd.concat([toprow, working], axis=0)
    
    # Add label names
    labels = []
    for item in working.index:
        if item in metric_name.index:
            add_label = metric_name.loc[item]['Name']
        else:
            add_label = item
        labels.append(add_label)
    working['Metric'] = labels
    
    # Put columns in correct order
    working = working[['Metric', 'Value'] + peer_group]
    return working



def update_spider_plot(n_clicks, ticker):
    """
    Create spider plot output
    """
    # GEt the data for the spider plot
    df_spider = get_spider_data(ticker)
    
    # Create the data for the spider chart
    num_vars = len(df_spider)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))
    values = df_spider['value']
    values = np.concatenate((values, [values[0]]))
    
    trace = go.Scatterpolar(
        r=values,
        theta=df_spider['labels'].tolist() + [df_spider['labels'].tolist()[0]],
        fill='toself',
        mode='lines+markers',
        marker=dict(size=8, color='#558BB8'),
        line=dict(color='#558BB8'),
        name='Data'
    )
    
    # Prepare the layout
    layout = go.Layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            ),
            angularaxis=dict(
                direction="clockwise",
                rotation=90
            )
        ),
        title={
            'text': 'Score Breakdown in Percentile vs Global Universe (higher better)',
        },
        showlegend=False
    )
    
    return {'data': [trace], 'layout': layout}



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



def dot_plot(ticker, metric1, metric2):
    """
    Creates a 2x2 dot plot for the country
    """
    min_mar_cap = 250
    min_ADV = 1

    if metric2 == 'BQ_score_chg_0y':
        yaxis_title='Business Quality Score Change (higher better)'
        title = 'Business Quality Score vs Business Quality Score Change'
    else:
        yaxis_title='Valuation Score (higher better)'
        title = 'Business Quality vs Valuation Scores'
    
    # Get sector and country for desired ticker
    sector = df.loc[ticker]['sector']
    exchange = df.loc[ticker]['exchange1']

    # Create a sub dataframe with the relevant data
    df_plot = df[df['exchange1'] == exchange]
    df_plot = df_plot[df_plot['mar_cap_mUSD'] > min_mar_cap]
    df_plot = df_plot[df_plot['3m_val_turnover_mUSD'] > min_ADV]
    df_plot.dropna(subset=['mar_cap_mUSD', '3m_val_turnover_mUSD'], inplace=True)
    df_plot['formatted_mar_cap_mUSD'] = df_plot['mar_cap_mUSD'].apply(lambda x: "{:,.0f}".format(x))
    df_plot['formatted_ADV_mUSD'] = df_plot['3m_val_turnover_mUSD'].apply(lambda x: "{:,.1f}".format(x))

    # Isolate sectors
    df_plot_sector = df_plot[df_plot['sector'] == sector]

    # hover text
    hover_texts_country = (df_plot.index + "<br>" + df_plot['name'] + "<br>Mar Cap (mUSD): " + df_plot['formatted_mar_cap_mUSD'] + 
                "<br>ADV (mUSD): " + df_plot['formatted_ADV_mUSD'] + "<br>Sector: " + df_plot['sector'] + "<br>Industry: " + df_plot['industry'])
    hover_texts_sector = (df_plot_sector.index + "<br>" + df_plot_sector['name'] + "<br>Mar Cap (mUSD): " + df_plot_sector['formatted_mar_cap_mUSD'] + 
                "<br>ADV (mUSD): " + df_plot_sector['formatted_ADV_mUSD'] + "<br>Sector: " + df_plot_sector['sector'] + "<br>Industry: " 
                + df_plot_sector['industry'])
    hover_text_ticker = (ticker + "<br>" + df_plot.loc[ticker]['name'] + "<br>Mar Cap (mUSD): " + df_plot.loc[ticker]['formatted_mar_cap_mUSD']+
                "<br>ADV (mUSD): " + df_plot.loc[ticker]['formatted_ADV_mUSD'] + "<br>Sector: " + df_plot.loc[ticker]['sector'] + "<br>Industry: " 
                + df_plot.loc[ticker]['industry'])


    # Create a blank figure
    fig = go.Figure()

    # Scatter plot for the country
    fig.add_trace(go.Scatter(x=df_plot[metric1],
                             y=df_plot[metric2],
                             mode='markers',
                             marker=dict(color='#9BB5CF'),
                             name=exchange,
                             hovertext=hover_texts_country,
                             hoverinfo='text'))

    # Scatter plot for the sector
    fig.add_trace(go.Scatter(x=df_plot_sector[metric1],
                             y=df_plot_sector[metric2],
                             mode='markers',
                             marker=dict(color='#DD8047'),
                             name=sector,
                             hovertext=hover_texts_sector,
                             hoverinfo='text',
                             textposition="top center"))

    # Scatter plot for the ticker
    fig.add_trace(go.Scatter(x=[df_plot.loc[ticker][metric1]],
                             y=[df_plot.loc[ticker][metric2]],
                             mode='markers+text',
                             marker=dict(color='red', size=10),
                             hovertext=hover_text_ticker,
                             hoverinfo='text',
                             name=ticker,
                             text=ticker,  # this will label the red dot with the ticker's name
                             textposition="top center"))

    # Update layout    
    fig.update_layout(
                 title=title,
                 xaxis_title='Business Quality Score (higher better)',
                 yaxis_title=yaxis_title,
                 xaxis=dict(gridcolor='gray', gridwidth=0.5, zerolinecolor='gray', zerolinewidth=0.5),
                 yaxis=dict(gridcolor='gray', gridwidth=0.5, zerolinecolor='gray', zerolinewidth=0.5),
                 plot_bgcolor='white',
                 paper_bgcolor='white',
                 height=700,
                 legend={
                     'x': 0.05,  # Adjust this value to change the horizontal position of the legend
                     'y': 0.95,  # Adjust this value to change the vertical position of the legend
                     'xanchor': 'left',  # Set the horizontal anchor to the left
                     'yanchor': 'top',   # Set the vertical anchor to the top
                     'bgcolor': 'rgba(255, 255, 255, 0.7)',  # Adjust the background color and opacity
                     'bordercolor': 'rgba(0, 0, 0, 0.3)',   # Adjust the border color and opacity
                     'borderwidth': 1,  # Set the border width
                 },
     )
    
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



def format_value(value):
    """
    Format the values column at the start of the datatable and returns a string
    """
    
    # Check if the value is NaN
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return ''
    # Check if the value is a number
    elif isinstance(value, (int, float)):
        formatted_value = '{:.2f}'.format(value)
        # Remove trailing zeros and the decimal point if the result is an integer
        formatted_value = formatted_value.rstrip('0').rstrip('.')
        return formatted_value
    else:
        return str(value)




# Load the datafile
df = pd.read_pickle('data/business_quality_data.pkl')
metric_name = pd.read_csv('inputs/metric_name.csv', index_col='Metric')


# Define the score historgrams for the first row
metric_list_A = ['BQ_score', 'V_score', 'EQ_score', 'govern_score']
graphs1=[f'histogram-{i+1}' for i in range(len(metric_list_A))]



# Define the histogram plots for the key metrics
#Capital Structure
met_roc = ['ROE', 'median ROE 3-yr', 'ROE_change', 'ROA', 'RNOA', 'ROCE']
g_roc = [f'hist_roc-{i+1}' for i in range(len(met_roc))]

#Dupont
met_dupont = ['PM', 'sales/assets', 'assets/equity']
g_dupont = [f'hist_dupont-{i+1}' for i in range(len(met_dupont))]

#Growth
met_growth = ['Revenue 1-yr CAGR', 'Revenue 3-yr CAGR', 'ShareholderEquity 1-yr CAGR', 'ShareholderEquity 3-yr CAGR']
g_growth = [f'hist_growth-{i+1}' for i in range(len(met_growth))]

#Profitability
met_profit = ['GPM%', 'OPMargin', 'OPM_change', 'PM', 'median PM 3-yr', 'PM_change', 'SGA_pctSales']
g_profit = [f'hist_profit-{i+1}' for i in range(len(met_profit))]

#Asset Intensity
met_ai = ['sales/assets', 'sales/asset_change', 'Sales/avg_NOA']
g_ai = [f'hist_ai-{i+1}' for i in range(len(met_ai))]

# Capital Structure
met_cap = ['assets/equity', 'assets/equity_change', 'net_debt/equity'] #Add net_debt
g_cap = [f'hist_cap-{i+1}' for i in range(len(met_cap))]

# Reinvestment
met_inv = ['R&D_pctSales'] #Add net_debt
g_inv = [f'hist_inv-{i+1}' for i in range(len(met_inv))]

# Shareholder Transaction
met_share = ['Share growth', 'div_payout_rate'] #Add net_debt
g_share = [f'hist_share-{i+1}' for i in range(len(met_share))]

# Valuation
met_val = ['P/E_ttm', 'P/B', 'P/S_ttm', 'EV/EBIT_ttm', 'EV/EBITDA_ttm']
g_val = [f'hist_val-{i+1}' for i in range(len(met_val))]

# Earnings Quality
met_eq = ['days_CCC', 'days_CCC_change', 'days_sales_outstanding', 'days_inventory_outstanding', 'days_payables_outstanding']
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
             dbc.Col(html.H1("Quality Company Overview"
                     , className="text-center")
                     , className="mb-5 mt-5", style={'padding-top':'20px'})
         ),
        
        # Get the ticker input and print the name
        dcc.Input(id='ticker-input', value='AAPL', type='text'),  # Default value is AAPL
        html.Button(id='submit-button', n_clicks=0, children='Submit'),

        # Give ticker name and some text
        dbc.Row(dbc.Col(html.H2(id='ticker-message'                     
                 , className="text-center")
                , className="mb-3 mt-3")),  # This will display the message style={'padding-top':'20px'}
        dbc.Row(dbc.Col(html.Div(id='stock-info'                     
                 , className="text-left")
                , className="mb-3 mt-3")),  # This will display the message style={'padding-top':'20px'}
        
        # Score 2x2 histogram 
        *generate_layout_children(graphs1, 2, 300),

        # Spider
        dcc.Graph(id='spider-plot'),
        
        # Dot plots
        dcc.Graph(id='dot-plot1'),
        
        dcc.Graph(id='dot-plot2', style={'height': '600px'}),
        
        # Historical line chart
        dcc.Graph(id='line-chart', style={'height': '450px'}),

        # SUMMARY TABLE
        dbc.Row(dbc.Col(html.H2("Summary Performance vs Peer Groups", className="text-left"), 
                         className="mb-3 mt-3", style={'padding-top':'20px'})),
        
        html.Div([
        dash_table.DataTable(id='summ-table',
                             style_table={'overflowY': 'auto'},
                             style_header = {
                                    'backgroundColor': '#3E5C7B',
                                    'color': 'white',
                                    'fontWeight': 'bold',
                                    'whiteSpace': 'normal'  # Enable text wrapping
                                }),
                ]),

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


# First callback for the histograms
@app.callback(
    [Output('histogram-1', 'figure'),
     Output('histogram-2', 'figure'),
     Output('histogram-3', 'figure'),
     Output('histogram-4', 'figure')],
    [Input('submit-button', 'n_clicks')],
    [dash.dependencies.State('ticker-input', 'value')]
)
def update_histograms(n_clicks, ticker):
    if ticker not in df.index:
        return [dash.no_update for _ in range(4)]  # Return no updates for all 4 graphs
    else:
        bins = 100
        return [generate_histogram(ticker, metric, bins) for metric in metric_list_A]


# Second callback for the spider plot and the message
@app.callback(
    [Output('spider-plot', 'figure'),
     Output('ticker-message', 'children'),
     Output('stock-info', 'children')],  # Two outputs: one for the plot and one for the message
    [Input('submit-button', 'n_clicks')],
    [dash.dependencies.State('ticker-input', 'value')]
)
def update_figure_and_message(n_clicks, ticker):
    if ticker not in df.index:
        return dash.no_update, 'Ticker not in database'  # No update to the graph, message updated
    else:
        fig = update_spider_plot(n_clicks, ticker)
        name = df.loc[ticker]['name'] + ' (' + ticker + ')'
        info = [f"Market Cap (bUSD):  {df.loc[ticker]['mar_cap_mUSD']/1000:,.1f}",
                html.Br(),
                f"ADV (mUSD): {df.loc[ticker]['3m_val_turnover_mUSD']:,.1f}",
                html.Br(),
                f"Sector: {df.loc[ticker]['sector']}",
                html.Br(),
                f"Industry: {df.loc[ticker]['industry']}"]
        return fig, name, info  # Graph updated, message updated with name


# Second callback for the spider plot and the message
@app.callback(
    [Output('dot-plot1', 'figure'),
     Output('dot-plot2', 'figure')],  # Two outputs: one for the plot and one for the message
    [Input('submit-button', 'n_clicks')],
    [dash.dependencies.State('ticker-input', 'value')]
)
def update_dot_plots(n_clicks, ticker):
    if ticker not in df.index:
        return dash.no_update, dash.no_update
    else:
        fig1 = dot_plot(ticker, 'BQ_score', 'V_score')
        fig2 = dot_plot(ticker, 'BQ_score', 'BQ_score_chg_0y')
    
        return fig1, fig2  # Graph updated, message updated with name


#Plots the historical percentile line chart
@app.callback(
    Output('line-chart', 'figure'),
    [Input('submit-button', 'n_clicks')],
    [dash.dependencies.State('ticker-input', 'value')]  # Assuming you have a ticker input field
)
def update_line_chart(n_clicks, ticker):
    if ticker not in df.index:
        return dash.no_update  # Handle the case when the ticker is not in your DataFrame
    
    # Select the historical data for the specified metrics
    historical_data = df.loc[ticker][['BQ_score_annual_0y_pctile', 'BQ_score_annual_-1y_pctile', 'BQ_score_annual_-2y_pctile']]*100
    
    # Create a line chart using Plotly
    fig = go.Figure()
    
    # Add traces for each metric:
    fig.add_trace(go.Scatter(
        x=['0 Year','-1 Year','-2 Year'],
        y=historical_data,
        mode='lines+markers',
    ))
    
    # Update the chart layout
    fig.update_layout(
        title='Historical BQ Score Percentiles',
        yaxis_title='Percentile',
        xaxis=dict(autorange='reversed')
    )
    
    # Change the line color for the single trace
    fig.update_traces(line=dict(color='#558BB8'))

    # Set the y-axis range from 0 to 1
    fig.update_yaxes(range=[0, 100])
    
    return fig



@app.callback(
    [Output('summ-table', 'data'),
     Output('summ-table', 'columns'),
     Output('summ-table', 'style_data_conditional')],
    [Input('submit-button', 'n_clicks')],
    [dash.dependencies.State('ticker-input', 'value')]
)
def update_table(n_clicks, ticker):
    if ticker not in df.index:
        return dash.no_update  # Return an empty figure if no ticker is provided

    # Get data
    data = table_data(ticker)
    data['Value'] = data['Value'].apply(format_value)

    # Define columns
    columns = [{'name': col, 'id': col} for col in data.columns]

    # Define the upper and lower thresholds and colours for the percentile columns in the chart
    threshold_upper = 85
    threshold_lower = 15
    colour_upper = '#B0CAC4'
    colour_lower = '#D08452'
    
    style_data_conditional = [
        # Set column relative widths
        {'if': {'column_id': 'Metric'},
         'width': '30%',
         'whiteSpace': 'normal'},  # Enable text wrapping
        {'if': {'column_id': 'Value'},
         'width': '14%'},
        {'if': {'column_id': 'Global'},
         'width': '14%'},
        {'if': {'column_id': 'All Country'},
         'width': '14%'},
        {'if': {'column_id': 'Country & Sector'},
         'width': '14%'},
        {'if': {'column_id': 'Global Industry'},
         'width': '14%'},
        
        #Highlight the outliers by row
        {'if': {'column_id': 'Global',
            'filter_query': f'{{Global}} > {threshold_upper}'},
         'backgroundColor': colour_upper},       
        {'if': {'column_id': 'Global',
            'filter_query': f'{{Global}} < {threshold_lower} && {{Global}} > 0'},
         'backgroundColor': colour_lower},          

        {'if': {'column_id': 'All Country',
            'filter_query': f'{{All Country}} > {threshold_upper}'},
         'backgroundColor': colour_upper},       
        {'if': {'column_id': 'All Country',
            'filter_query': f'{{All Country}} < {threshold_lower} && {{All Country}} > 0'},
         'backgroundColor': colour_lower}, 
        
        {'if': {'column_id': 'Country & Sector',
            'filter_query': f'{{Country & Sector}} > {threshold_upper}'},
         'backgroundColor': colour_upper},       
        {'if': {'column_id': 'Country & Sector',
            'filter_query': f'{{Country & Sector}} < {threshold_lower} && {{Country & Sector}} > 0'},
         'backgroundColor': colour_lower},
        
        {'if': {'column_id': 'Global Industry',
            'filter_query': f'{{Global Industry}} > {threshold_upper}'},
         'backgroundColor': colour_upper},       
        {'if': {'column_id': 'Global Industry',
            'filter_query': f'{{Global Industry}} < {threshold_lower} && {{Global Industry}} > 0'},
         'backgroundColor': colour_lower},
        
        # Ensure first row is white regardless
        {'if': {'row_index': 0},
         'backgroundColor': 'white'}
        
        ]
    
    
    

    return data.to_dict('records'), columns, style_data_conditional



# Third callback for all the histograms
output_args = [Output(graph, 'figure') for graph in g_comb]
@app.callback(
    output_args,
    [Input('submit-button', 'n_clicks')],
    [dash.dependencies.State('ticker-input', 'value')]
)
def update_histograms2(n_clicks, ticker):
    if ticker not in df.index:
        return [dash.no_update for _ in range(len(g_comb))]  # Return no updates based on the length of graphs2
    else:
        bins = 100  # Modify as needed
        return [generate_histogram(ticker, metric, bins) for metric in (met_comb)]



# needed only if running this as a single page app
# =============================================================================
# if __name__ == '__main__':
#     app.run_server(debug=True)
# =============================================================================

