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
from dash import dash_table

from dash.dependencies import Input, Output

from dash.dash_table.Format import Format, Scheme
import dash_bootstrap_components as dbc

from app import app

# needed only if running this as a single page app
# external_stylesheets = [dbc.themes.LUX]
# =============================================================================
# app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
# =============================================================================


# Set initial variables
axes_scale = 'log' # can be either 'linear', or 'log'
country = 'GLOBAL'

# Import the data and create the table df
df = pd.read_pickle('data/company_info_short.pkl')
# If loading this page only need to look in directory ../data...

df_table = df[['name','industry_comb','country_name','mar_cap_mUSD','3m_val_turnover_mUSD']]
#df_table.reset_index(inplace=True)
#df_table.reset_index(inplace=True)
#print(df_table)
#df_table['level_0'] += 1

df_table = df_table.copy()
df_table['level_0'] = range(1, len(df) + 1)
df_table['mar_cap_mUSD'] = df_table['mar_cap_mUSD']/1000
table_rows = 50

# Create the dropdown for the table
country_dropdown = [{'label':'GLOBAL', 'value': 'GLOBAL'}]
for name in df['country_name'].unique():
    country_dropdown.append({'label': name, 'value': name})

# Setup the column formatting for the table
columns = [
    dict(id='level_0', name='Global Rank'),
    dict(id='Symbol', name='Ticker'),
    dict(id='name', name='Company Name'),
    dict(id='industry_comb', name='Industry'),
    dict(id='country_name', name='Country'),
    dict(id='mar_cap_mUSD', name='Market Cap, $b', type='numeric', format=Format(precision=1, scheme=Scheme.fixed).group(True)),
    dict(id='3m_val_turnover_mUSD', name='3m ADV, $m', type='numeric', format=Format(precision=0, scheme=Scheme.fixed).group(True)),
]

# only for standalone use
# =============================================================================
# app = dash.Dash()
# server = app.server
# =============================================================================


# change to app.layout if running as single page app instead
layout = html.Div([
     dbc.Container([
        dbc.Row(
            dbc.Col(html.H1("Overview of Investible Universe"
                    , className="text-center")
                    , className="mb-5 mt-5", style={'padding-top':'20px'})
        ),

        dbc.Row([
            dbc.Col(html.P(children=['All globally listed equities with 3m ADV >$1m and market cap >$10m', 
                  html.Br(), 
                  'Multi-listed shares shown where ADV of secondary listing >$50m',
                  html.Br(), 
                  'Spot data last updated as of 17 June 2021',
                  html.Br(), 
                  'Total number of listed equities: ' + str(len(df.index.unique()))
                  ])  , className="mb-5")
        ]),

        dbc.Row([
            dbc.Col(html.P('Axes scale type :'), width=2, className="mb-4"),
            dbc.Col(
                dcc.Dropdown(id='axes_type',
                     options=[{'label': 'linear', 'value': 'linear'},
                              {'label': 'logarithmic', 'value': 'log'}],
                     value='log',
                     placeholder = 'Select axes type'),
                        width=4, className="mb-4")
        ]),

        dbc.Row(
            dbc.Col(html.H4([html.Br(), 
                             'Chart of all listed companies by Market Cap and ADV']
                    , className="text-center")
              #      , className="mb-5 mt-5")
            )),

        dcc.Graph(id='scatter1'),
    
        dbc.Row(
            dbc.Col(html.H4(' '
                    , className="text-center")
                    , className="mb-5 mt-5")
            ),
    
       dbc.Row([
            dbc.Col(html.P('Select country: '), width=2, className="mb-4"),
            dbc.Col(
                dcc.Dropdown(id='country_selector',
                 options=country_dropdown,
                 value='GLOBAL'),
                        width=4, className="mb-4")
        ]),

        dbc.Row(
            dbc.Col(html.H4([html.Br(), 
                             'Largest companies by country']
                    , className="text-center")
              #      , className="mb-5 mt-5")
            )),

        html.Div(id='table1'),    

        dbc.Row(
            dbc.Col(html.H3(' '
                    , className="text-center")
                    , className="mb-5 mt-5"))        
        
    ]), 
])


# Control for the scatter plot
@app.callback(Output('scatter1', 'figure'),
              [Input('axes_type', 'value')])

def update_figure(selected_scale):
    traces = []
    for name in df['country_name'].unique():
        df_group = df[df['country_name'] == name]
        traces.append(go.Scatter(
            x=df_group['mar_cap_mUSD'],
            y=df_group['3m_val_turnover_mUSD'],
            text = df_group.index + '<br>' + df_group.name + '<br>' + 'Sector: ' + df_group.sector_comb
                + '<br>' + 'Industry: ' + df_group.industry_comb + '<br>' + 'Country based: ' + df_group.country_name,
            mode='markers',
            marker = dict(      # change the marker style
                    size = 8,
                    symbol = 'circle',
                    line = dict(
                        width = 1)
                    ),
            name=name)
        )
     
    return {
        'data': traces,
        'layout': go.Layout(
                # title = '<b>Chart of all listed companies by Market Cap and ADV</b>', # Graph title
                xaxis = dict(title = 'Market Cap, m USD'), # x-axis label
                yaxis = dict(title = 'ADV, m USD'), # y-axis label,
                xaxis_type = selected_scale,
                yaxis_type = selected_scale,
                hovermode ='closest', # handles multiple points landing on the same vertical
                height = 1000,# px
                template = 'seaborn',
                font_family = 'Nunito Sans',
                margin=dict(t=10)
                )
    }

# Controls for the table
@app.callback(Output('table1', 'children'),
              [Input('country_selector', 'value')])

def table_data(country):
    if country == 'GLOBAL':
        df_table2 = df_table.iloc[0:table_rows]
    else:
        df_table2 = df_table[df_table['country_name'] == country]
        df_table2 = df_table2.iloc[0:table_rows]
     

     
    df_table2.reset_index(inplace=True)
    
    # print(df_table2)
    
    data = df_table2.to_dict('records')

    
    cell_format = {'fontFamily': 'Nunito Sans', 'text-align': 'left', 'whiteSpace': 'normal', 'height': 'auto', 'padding':'5px'}
    style_header={'fontWeight': 'bold', 'backgroundColor': '#343a40', 'color': 'white'}
    
    return html.Div([dash_table.DataTable(data = data, columns = columns, style_cell = cell_format, style_header = style_header)])


# needed only if running this as a single page app
# =============================================================================
# if __name__ == '__main__':
#     app.run_server()
# =============================================================================
