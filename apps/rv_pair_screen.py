#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 09:21:43 2021

@author: darylspelman
"""

import pandas as pd
import plotly.graph_objs as go

from dash import dcc
from dash import html

from dash.dependencies import Input, Output
from dash.dash_table.Format import Format, Scheme, Trim, Sign
from dash.dash_table import DataTable, FormatTemplate
import dash_bootstrap_components as dbc

from app import app

# needed only if running this as a single page app
# external_stylesheets = [dbc.themes.LUX]
# =============================================================================
# app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
# =============================================================================


# only for standalone use
# =============================================================================
# app = dash.Dash()
# server = app.server
# =============================================================================

options=[{'label': '1m z-score', 'value': '1m_z-score'},
         {'label': '2m z-score', 'value': '2m_z-score'},
         {'label': '3m z-score', 'value': '3m_z-score'},         
         {'label': '6m z-score', 'value': '6m_z-score'},          
         {'label': '12m z-score', 'value': '12m_z-score'},
         {'label': '2y z-score', 'value': '2y_z-score'},
         {'label': '5y z-score', 'value': '5y_z-score'},
         {'label': 'z-score compound metric 1', 'value': 'z_score_comp1'},
         {'label': 'z-score compound metric 2', 'value': 'z_score_comp2'},
         {'label': 'Premium/ Discount', 'value': 'gap_to_fair'},
         {'label': 'percentile', 'value': 'percentile'}
         ]

# Load and clean the screening data
screen = pd.read_pickle('data/rv_pair_screening_stats.pkl')
# Remove the Nans in the text columns so the putput displays properly
fix_cols = ['name_ticker1', 'name_ticker2']
for col in fix_cols:
    screen[col] = screen[col].fillna('n/a')

# Create the pair types for first dropdown
type_list = screen['type'].unique()
options2 = [{"label": i, "value": i} for i in type_list]
options2.append({"label":'ALL',"value":'ALL'})

fig1 = go.Figure()
cols = ['1m_z-score', '2m_z-score', '3m_z-score', '6m_z-score', '12m_z-score', '2y_z-score', '5y_z-score']
data=screen
# Calculate content for z-score boxplots
for col in cols:
    fig1.add_trace(go.Box(x=data[col], 
                         name=col,
                         width=0.4,
                         boxpoints='all',
                         jitter=0.5,
                         marker_size=5,
                         marker = dict(outliercolor='red',          
                                        line=dict(color='gray',
                                                  width=1,
                                            outliercolor='red',
                                        outlierwidth=2)),
                        text=data.index + '<br>' + '<br>' + 
                            'Ticker 1: ' + data.ticker1 + '<br>' + data.name_ticker1 + '<br>' +
                            'Market Cap bUSD: ' + data.marcap_mUSD_ticker1.apply(lambda x: '%.1f' %(x/1000)) + '<br>' +
                            'ADV mUSD: ' + data.ADV_mUSD_ticker1.apply(lambda x: '%.1f' %(x))+ '<br>' + '<br>' + 
                            'Ticker 2: ' + data.ticker2 + '<br>' + data.name_ticker2 + '<br>' +
                            'Market Cap bUSD: ' + data.marcap_mUSD_ticker2.apply(lambda x: '%.1f' %(x/1000)) + '<br>' +
                            'ADV mUSD: ' + data.ADV_mUSD_ticker2.apply(lambda x: '%.1f' %(x))  
                        ))
fig1.update_layout(title='Z-score boxplots', title_x=0.5,
               xaxis = dict(title = 'z-score', zeroline=True, zerolinecolor='black'),
               hovermode ='closest',
               height=800)


# Calculate content for compound z-score boxplot
fig2 = go.Figure()
cols = ['z_score_comp1', 'z_score_comp2']
data=screen
for col in cols:
    fig2.add_trace(go.Box(x=data[col], 
                         name=col,
                         width=0.4,
                         boxpoints='all',
                         jitter=0.5,
                         marker_size=5,
                         marker = dict(outliercolor='red',          
                                        line=dict(color='gray',
                                                  width=1,
                                            outliercolor='red',
                                        outlierwidth=2)),
                        text=data.index + '<br>' + '<br>' + 
                            'Ticker 1: ' + data.ticker1 + '<br>' + data.name_ticker1 + '<br>' +
                            'Market Cap bUSD: ' + data.marcap_mUSD_ticker1.apply(lambda x: '%.1f' %(x/1000)) + '<br>' +
                            'ADV mUSD: ' + data.ADV_mUSD_ticker1.apply(lambda x: '%.1f' %(x))+ '<br>' + '<br>' + 
                            'Ticker 2: ' + data.ticker2 + '<br>' + data.name_ticker2 + '<br>' +
                            'Market Cap bUSD: ' + data.marcap_mUSD_ticker2.apply(lambda x: '%.1f' %(x/1000)) + '<br>' +
                            'ADV mUSD: ' + data.ADV_mUSD_ticker2.apply(lambda x: '%.1f' %(x))  
                        ))
fig2.update_layout(title='Compound z-score boxplots', title_x=0.5,
               xaxis = dict(title = 'z-score'),
               hovermode ='closest',
               height=350)
    

# Calculate data for z-score vs percentile
fig3 = go.Figure()
for item in screen['type'].unique():
    data=screen[screen['type']==item]
    fig3.add_trace(go.Scatter(
        x=data['percentile'],
        y=data['z_score_comp2'],
        mode='markers',
        name=item,
        text = data.index + '<br>' + '<br>' + 
                'Ticker 1: ' + data.ticker1 + '<br>' + data.name_ticker1 + '<br>' +
                'Market Cap bUSD: ' + data.marcap_mUSD_ticker1.apply(lambda x: '%.1f' %(x/1000)) + '<br>' +
                'ADV mUSD: ' + data.ADV_mUSD_ticker1.apply(lambda x: '%.1f' %(x))+ '<br>' + '<br>' + 
                'Ticker 2: ' + data.ticker2 + '<br>' + data.name_ticker2 + '<br>' +
                'Market Cap bUSD: ' + data.marcap_mUSD_ticker2.apply(lambda x: '%.1f' %(x/1000)) + '<br>' +
                'ADV mUSD: ' + data.ADV_mUSD_ticker2.apply(lambda x: '%.1f' %(x))  
    ))
# Set options common to all traces with fig.update_traces
fig3.update_traces(mode='markers', marker_size=7)
fig3.update_layout(title='Compound z-score metric vs ratio percentile', title_x=0.5,
               xaxis = dict(title = 'Percentile'),
               yaxis = dict(title = 'Compound-z-score'),
               hovermode ='closest',
               height=700)


fig4 = go.Figure()
for item in screen['type'].unique():
    data=screen[screen['type']==item]
    fig4.add_trace(go.Scatter(
        x=data['percentile'],
        y=data['gap_to_fair'],
        mode='markers',
        name=item,
        text = data.index + '<br>' + '<br>' + 
                'Ticker 1: ' + data.ticker1 + '<br>' + data.name_ticker1 + '<br>' +
                'Market Cap bUSD: ' + data.marcap_mUSD_ticker1.apply(lambda x: '%.1f' %(x/1000)) + '<br>' +
                'ADV mUSD: ' + data.ADV_mUSD_ticker1.apply(lambda x: '%.1f' %(x))+ '<br>' + '<br>' + 
                'Ticker 2: ' + data.ticker2 + '<br>' + data.name_ticker2 + '<br>' +
                'Market Cap bUSD: ' + data.marcap_mUSD_ticker2.apply(lambda x: '%.1f' %(x/1000)) + '<br>' +
                'ADV mUSD: ' + data.ADV_mUSD_ticker2.apply(lambda x: '%.1f' %(x))  
    ))
# Set options common to all traces with fig.update_traces
fig4.update_traces(mode='markers', marker_size=7)
fig4.update_layout(title='Spread Premium vs Ratio Percentile', title_x=0.5,
               xaxis = dict(title = 'Percentile'),
               yaxis = dict(title = 'Premium/Discount', tickformat = ',.0%', zeroline=True, zerolinecolor='black'),
               hovermode ='closest',
               height=700)


# Premium/Discountrank chart
fig5 = go.Figure()
screen_temp = screen.copy()
screen_temp = screen_temp.sort_values(by=['gap_to_fair'])
for item in screen_temp['type'].unique():
    data=screen_temp[screen_temp['type']==item]
    fig5.add_trace(go.Scatter(
        x=data['gap_to_fair'],
        y=data.index,
        mode='markers',
        name=item,
        text = data.index + '<br>' + '<br>' + 
                'Ticker 1: ' + data.ticker1 + '<br>' + data.name_ticker1 + '<br>' +
                'Market Cap bUSD: ' + data.marcap_mUSD_ticker1.apply(lambda x: '%.1f' %(x/1000)) + '<br>' +
                'ADV mUSD: ' + data.ADV_mUSD_ticker1.apply(lambda x: '%.1f' %(x))+ '<br>' + '<br>' + 
                'Ticker 2: ' + data.ticker2 + '<br>' + data.name_ticker2 + '<br>' +
                'Market Cap bUSD: ' + data.marcap_mUSD_ticker2.apply(lambda x: '%.1f' %(x/1000)) + '<br>' +
                'ADV mUSD: ' + data.ADV_mUSD_ticker2.apply(lambda x: '%.1f' %(x))  
    ))
# Set options common to all traces with fig.update_traces
fig5.update_traces(mode='markers', marker_size=7)
fig5.update_layout(title='Premium/Discount Rank', title_x=0.5,
              xaxis = dict(title = 'Premium/Discount', tickformat = ',.0%', zeroline=True, zerolinecolor='black'),
              yaxis = dict(title = 'Average Premium', tickformat = ',.0%', zeroline=True, zerolinecolor='black'),
  #             xaxis = {'zeroline':True, 'zerolinecolor': 'black'})               
              hovermode ='closest',
              height=700)





### Calculate historical summary spreads
df = pd.read_pickle('data/rv_pair_summary_history.pkl')

# Get the list of spread names
col_list = df.columns
spreads = []
for item in col_list:
    if 'median' in item:
        spreads.append(item.replace('_median',''))

# Calculate the min number of datapoints per row acceptable for displaying data and return a dictionary
# Helps to eliminate outliers
min_level = 0.55
min_num={}
for spread in spreads:
    min_c = df['{}_count'.format(spread)].max() * min_level
    min_num[spread] = min_c

# Calculate the rolling average and put the output data into new summary dataframe
df_chart = pd.DataFrame()
roll_days = 20
for spread in spreads:
    temp = pd.DataFrame()
    temp['{}'.format(spread)] = df['{}_mean'.format(spread)]
    temp['{}'.format(spread)] = temp['{}'.format(spread)][df['{}_count'.format(spread)] > min_num[spread]]
    temp.dropna(inplace=True)
    temp['{}_rollavg'.format(spread)]=temp['{}'.format(spread)].rolling(window=roll_days).mean()
    df_chart = pd.concat([df_chart,temp], axis=1)

# Plot 3m rolling average over time
df_chart.sort_index(inplace=True)
fig6 = go.Figure()
for spread in spreads:
    fig6.add_trace(go.Scatter(x=df_chart.index, 
                             y=df_chart['{}_rollavg'.format(spread)], 
                             mode='lines',
                             line={'width':2},
                             marker={'size':2},
                             name=spread))
fig6.update_layout(title='Mean spread by Arbitrage type (1-month rolling average)', 
                  title_x=0.5, 
                  height=700,
                  yaxis = dict(title = 'Average Premium', tickformat = ',.0%', zeroline=True, zerolinecolor='black'),
#                  xaxis = {'zeroline':True, 'zerolinecolor': 'black'}
                )

percentage2 = FormatTemplate.percentage(2)
percentage = FormatTemplate.percentage(1)
columns = [
    dict(id='index', name='Ticker'),
    dict(id='name_ticker1', name='Name'),
    dict(id='type', name='Type'),
    dict(id='ratio', name='Ratio', type='numeric', format=Format(precision=2, scheme=Scheme.fixed, trim=Trim.yes).group(True)),
    dict(id='gap_to_fair', name='Pre-mium', type='numeric', format=Format(precision=0, scheme=Scheme.percentage, sign=Sign.positive)),
    dict(id='percentile', name='Per-cent-ile', type='numeric', format=Format(precision=0, scheme=Scheme.fixed, trim=Trim.yes).group(True)),
    dict(id='1m_z-score', name='1m zscr', type='numeric', format=Format(precision=1, scheme=Scheme.fixed, trim=Trim.yes).group(True)),
    dict(id='2m_z-score', name='2m zscr', type='numeric', format=Format(precision=1, scheme=Scheme.fixed, trim=Trim.yes).group(True)),
    dict(id='3m_z-score', name='3m zscr', type='numeric', format=Format(precision=1, scheme=Scheme.fixed, trim=Trim.yes).group(True)),
    dict(id='6m_z-score', name='6m zscr', type='numeric', format=Format(precision=1, scheme=Scheme.fixed, trim=Trim.yes).group(True)),
    dict(id='12m_z-score', name='1y zscr', type='numeric', format=Format(precision=1, scheme=Scheme.fixed, trim=Trim.yes).group(True)),
    dict(id='2y_z-score', name='1y zscr', type='numeric', format=Format(precision=1, scheme=Scheme.fixed, trim=Trim.yes).group(True)),
    dict(id='5y_z-score', name='1y zscr', type='numeric', format=Format(precision=1, scheme=Scheme.fixed, trim=Trim.yes).group(True)),
    dict(id='z_score_comp1', name='cmp1 zscr', type='numeric', format=Format(precision=1, scheme=Scheme.fixed, trim=Trim.yes).group(True)),
    dict(id='z_score_comp2', name='cmp2 zscr', type='numeric', format=Format(precision=1, scheme=Scheme.fixed, trim=Trim.yes).group(True)),
]

# Other table formatting
cell_format = {'fontFamily': 'Nunito Sans', 'text-align': 'left', 'whiteSpace': 'normal', 'height': 'auto', 'padding':'5px'}
style_header={'fontWeight': 'bold', 'backgroundColor': '#343a40', 'color': 'white'} 


# change to app.layout if running as single page app instead
layout = html.Div([
     dbc.Container([
        dbc.Row(
            dbc.Col(html.H1("RV Pair Screen"
                    , className="text-center")
                    , className="mb-5 mt-5", style={'padding-top':'20px'})
        ),
       
       dbc.Row(
            dbc.Col(dcc.Graph(id='scr_rank', figure=fig6)
                    , className="mb-3", width=12
        )),

       dbc.Row(
            dbc.Col(dcc.Graph(id='zscore_box2', figure=fig2)
                    , className="mb-3", width=12
        )),

       dbc.Row(
            dbc.Col(dcc.Graph(id='scr_scatter1', figure=fig3)
                    , className="mb-3", width=12
        )),

       dbc.Row(
            dbc.Col(dcc.Graph(id='zscore_box1', figure=fig1)
                    , className="mb-3", width=12
        )),

       dbc.Row(
            dbc.Col(dcc.Graph(id='scr_scatter2', figure=fig4)
                    , className="mb-3", width=12
        )),

       dbc.Row(
            dbc.Col(dcc.Graph(id='scr_rank', figure=fig5)
                    , className="mb-3", width=12
        )),

        dbc.Row([
            dbc.Col(html.P('Select column to rank: '), width=2),
            dbc.Col(dcc.Dropdown(
                id='cat_dropdown',
                options=options,
                value='z_score_comp2'
                ), className="mb-3", width=4),
            dbc.Col(html.P('Select spread type: '), width=2),
            dbc.Col(dcc.Dropdown(
                id='type_dropdown',
                options=options2,
                value='ALL'
                ), className="mb-3", width=4),
            ]),

        dbc.Row([dbc.Col(
                    DataTable(id='pair_table', 
                           columns = columns, 
                           style_cell = cell_format, 
                           style_header = style_header),
                           className="mb-5", width=12)
                ]), 
        ])
])


@app.callback(Output('pair_table', 'data'),
              [Input('cat_dropdown', 'value'),
               Input('type_dropdown', 'value')])
def table_data_sort(sort_col,type_val):
    sort_data = screen.reset_index()
    sort_data['sort_col']=sort_data[sort_col].abs()
    sort_data.sort_values(by='sort_col', ascending=False, inplace=True)
    if type_val != 'ALL':
        sort_data = sort_data[sort_data['type']==type_val]
    return sort_data.to_dict('records')



# needed only if running this as a single page app
# =============================================================================
# if __name__ == '__main__':
#     app.run_server()
# =============================================================================
