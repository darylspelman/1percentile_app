# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 20:01:03 2023

@author: daryl
"""

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from scipy.stats import percentileofscore

#Needed when running in full environment
from app import app

#from memory_profiler import profile



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
            'EQ_accruals_comp_score_percentile',
            'EQ_assumption_comp_score_percentile',
            'EQ_risk_comp_score_percentile',
            'EQ_compound_comp_score_percentile',
            'V_score_fundamental_percentile',
            'V_score_relative_percentile',
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
                            'EQ:\nAccruals',
                            'EQ:\nAccounting Assumptions',
                            'EQ:\nRisk Indicator',
                            'EQ:\nCompound Metric',
                            'Valuation:\nFundamental',
                            'Valuation:\nRelative',
                            'Governance:\nAlignment with Shareholder',
                            'Governance:\nBoard Quality',
                            'Governance:\nCommittee Quality']
    return df_spider



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
        marker=dict(size=8, color='#3E5C7B'),
        line=dict(color='#3E5C7B'),
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
        height_val=200

    fig = go.Figure()
    # Compute histogram using numpy first to get the bin counts
    hist, bin_edges = np.histogram(data, bins=bins)
    max_y = np.max(hist)  # Corrected to calculate the maximum value in hist
    
    hist_trace = go.Histogram(x=data, name=metric, marker=dict(color='blue', opacity=1), nbinsx=bins)
    fig.add_trace(hist_trace)

    if np.isnan(specific_value) != True:
        fig.add_shape(go.layout.Shape(type='line', x0=specific_value, x1=specific_value, y0=0, y1=max_y, line=dict(color='red', width=2)))
        fig.add_annotation(text=custom_text, xref="paper", yref="paper", x=0.05, y=0.95, showarrow=False, align="left", bgcolor="white", bordercolor="black", borderwidth=1, font=dict(size=12))
    
    fig.update_layout(title={'text': metric, 'y':0.95, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'},
                      height = height_val,
                      margin=dict(l=20, r=20, t=50, b=20))
    
    # Set a fixed y-axis range to ensure consistency
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

    # Isolate sectors
    df_plot_sector = df_plot[df_plot['sector'] == sector]

    # hover text
    hover_texts_country = df_plot.index + "<br>" + df_plot['name'] + "<br>Mar Cap (mUSD): " + df_plot['formatted_mar_cap_mUSD']
    hover_texts_sector = df_plot_sector.index + "<br>" + df_plot_sector['name'] + "<br>Mar Cap (mUSD): " + df_plot_sector['formatted_mar_cap_mUSD']
    hover_text_ticker = ticker + "<br>" + df_plot.loc[ticker]['name'] + "<br>Mar Cap (mUSD): " + df_plot.loc[ticker]['formatted_mar_cap_mUSD']

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
    fig.update_layout(title=title,
                      xaxis_title='Business Quality Score (higher better)',
                      yaxis_title=yaxis_title,
                      xaxis=dict(gridcolor='gray', gridwidth=0.5, zerolinecolor='gray', zerolinewidth=0.5),
                      yaxis=dict(gridcolor='gray', gridwidth=0.5, zerolinecolor='gray', zerolinewidth=0.5),
                      plot_bgcolor='white',
                      paper_bgcolor='white',
                      height=600
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



# Load the datafile
df = pd.read_pickle('data/business_quality_data.pkl')

# Define the score historgrams for the first row
metric_list1 = ['BQ_score', 'V_score', 'EQ_score', 'govern_score']
graphs1=[f'histogram-{i+1}' for i in range(len(metric_list1))]

# Define the histogram plots for the key metrics
metric_list2 = ['ROE', 'median ROE 3-yr', 'ROE_change', 'ROA', 'RNOA', 'assets/equity', 'assets/equity_change', 'OPMargin', 
               'PM', 'median PM 3-yr', 'PM_change', 'OPM_change', 'Revenue 1-yr CAGR', 'ShareholderEquity 1-yr CAGR', 
               'Revenue 3-yr CAGR', 'ShareholderEquity 3-yr CAGR', 'Share growth', 'div_payout_rate', 
               'sales/assets', 'sales/asset_change', 'Sales/avg_NOA', 'days_CCC', 'days_CCC_change', 'R&D_pctSales',
               'days_sales_outstanding', 'days_inventory_outstanding', 'days_payables_outstanding',
                'P/E_ttm', 'P/B', 'P/S_ttm', 'EV/EBIT_ttm', 'EV/EBITDA_ttm']
graphs2 = [f'hist-{i+1}' for i in range(len(metric_list2))]


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
        dbc.Row(dbc.Col(html.H2(id='ticker-message'                     
                 , className="text-center")
                , className="mb-3 mt-3")),  # This will display the message style={'padding-top':'20px'}
        
        
        *generate_layout_children(graphs1, 4, 300),  # 3 charts per line as an example
        
        dcc.Graph(id='spider-plot'),
        dcc.Graph(id='dot-plot1'),
        dcc.Graph(id='dot-plot2'),
        *generate_layout_children(graphs2, 4, 200),  # 3 charts per line as an example
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
        return [generate_histogram(ticker, metric, bins) for metric in metric_list1]


# Second callback for the spider plot and the message
@app.callback(
    [Output('spider-plot', 'figure'),
     Output('ticker-message', 'children')],  # Two outputs: one for the plot and one for the message
    [Input('submit-button', 'n_clicks')],
    [dash.dependencies.State('ticker-input', 'value')]
)
def update_figure_and_message(n_clicks, ticker):
    if ticker not in df.index:
        return dash.no_update, 'Ticker not in database'  # No update to the graph, message updated
    else:
        fig = update_spider_plot(n_clicks, ticker)
        name = df.loc[ticker]['name'] + ' (' + ticker + ')'
        return fig, name  # Graph updated, message updated with name

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


# Third callback for all the histograms
output_args = [Output(graph, 'figure') for graph in graphs2]
@app.callback(
    output_args,
    [Input('submit-button', 'n_clicks')],
    [dash.dependencies.State('ticker-input', 'value')]
)
def update_histograms2(n_clicks, ticker):
    if ticker not in df.index:
        return [dash.no_update for _ in range(len(graphs2))]  # Return no updates based on the length of graphs2
    else:
        bins = 75  # Modify as needed
        return [generate_histogram(ticker, metric, bins) for metric in metric_list2]



# needed only if running this as a single page app
# =============================================================================
# if __name__ == '__main__':
#     app.run_server(debug=True)
# =============================================================================

