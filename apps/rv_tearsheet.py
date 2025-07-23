#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 09:21:43 2021

@author: darylspelman
"""

import pandas as pd
from scipy import stats


import yfinance as yf


from dash import dcc
from dash import html

import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.figure_factory as ff
from dash.dash_table import DataTable
from dash.dash_table.Format import Format
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate


# Needed if run as part of multi-page dashboard
from app import app




# needed only if running this as a single page app
# =============================================================================
# app = dash.Dash()
# =============================================================================


def get_rv_stats_tearsheet(ticker1, ticker2, fair_val, rv_type, df):
    """
    Takes ticker pairs and the fair value ratio
    Returns a dataframe with historical trend lines and second summary frame for screening file
    """

    summ = pd.DataFrame()
    summ2 = pd.DataFrame()
    
    #Calculate fair val
    df['gap_to_fair'] = df['ratio']/fair_val - 1

    # Cycle through all the time periods and calculate the required z-scores
    # periods = {'1m':20, '2m':40, '3m':60, '6m':125, '12m':250, '2y': 500, '5y': 1250}
    for key in periods.keys():
        time = key
        length = periods[key]

        # Calc avg + st dev
        df['{}_avg'.format(time)]=df['ratio'].rolling(window=length).mean()
        df['{}_std'.format(time)]=df['ratio'].rolling(window=length).std()
        
        # Calculate correlations
        df['{}_corr'.format(time)]=df[ticker1].rolling(window=length).corr(df[ticker2])

        # Calculate -3 to +3 stdev lines
        for i in [-3,-2,-1,1,2,3]:
            if i<0:
                label=str(i)
            else:
                label='+' + str(i)      
            df['{}_{}std'.format(time,label)]=df['{}_avg'.format(time)] + i * df['{}_std'.format(time)]

        # Calculate z-score
        df['{}_z-score'.format(time)]=(df['ratio']-df['{}_avg'.format(time)])/df['{}_std'.format(time)]
    
    # Calculate corresponding summary table for screening
    label = ticker1+'/'+ticker2
    
    summ[label] = df.iloc[-1]
    summ.loc['type'] = rv_type
    summ2 = summ.loc[['type','ratio','gap_to_fair']]
    summ2.loc['percentile'] = stats.percentileofscore(df['ratio'], df['ratio'][-1])
    summ2 = pd.concat([summ2, summ.loc[['3m_corr', '6m_corr', '12m_corr', '2y_corr','1m_z-score','2m_z-score','3m_z-score','6m_z-score','12m_z-score','2y_z-score','5y_z-score']]])

    # summ2 = summ2.append(summ.loc[['3m_corr', '6m_corr', '12m_corr', '2y_corr','1m_z-score','2m_z-score','3m_z-score','6m_z-score','12m_z-score','2y_z-score','5y_z-score']])
    
    return df, summ2


                                 
# Loads the list of metrics to be sumamrised in the dashboard
pair_list = pd.read_csv('inputs/rv.csv')
pair_list['pair_name']=pair_list['ticker1_y']+'/'+pair_list['ticker2_y']
curr_table = pd.read_pickle('data/currency_scrape.pkl')


# Create the pair types for first dropdown
type_list = pair_list['type'].unique()
option1 = [{"label": i, "value": i} for i in type_list]

# Inputs for the time dropdown
period_choice = [
#                 {'label':'1 month', 'value': '1mo'},
#                 {'label':'3 months', 'value': '3mo'},
#                 {'label':'6 months', 'value': '6mo'},
                 {'label':'1 year', 'value': '1y'},
                 {'label':'2 years', 'value': '2y'},
                 {'label':'5 years', 'value': '5y'},
                 {'label':'10 years', 'value': '10y'},
#                 {'label':'YTD', 'value': 'ytd'},
                 {'label':'max', 'value': 'max'}]  

# Used for calculating z-scores
periods = {'1m':20, '2m':40, '3m':60, '6m':125, '12m':250, '2y': 500, '5y': 1250}






# change to app.layout if running as single page app instead
layout = html.Div([    
    dbc.Container([
        dbc.Row(
            dbc.Col(html.H1(id='title', children="RV Pair Tearsheet"
                    , className="text-center")
                    , className="mb-5 mt-5", style={'padding-top':'20px'})
                ),

        dbc.Row([
            dbc.Col(dcc.Dropdown(
                id='cat_dropdown',
                options=option1,
                value=type_list[0]
                ),width=4), 

            dbc.Col(dcc.Dropdown(
                id='pair_dropdown',
                placeholder='Select the RV pair to review'
                ),width=4), 
                       
            dbc.Col(dcc.Dropdown(
                id='time_dropdown',
                options=period_choice,
                value='10y'
                ),width=2),
            
           dbc.Col(dbc.Button("Load Data", id='rv_data_button', n_clicks = 0, 
                              color="primary", active = True, style={'width':'100%'}), align = 'center', width=2)           
            
            ], className="mb-5"),

        html.Div(id='status_summ', className="mb-5"), 
        html.Div(id='rv_1', className="mb-5"),        
        ]),
    ])


@app.callback(
    Output('pair_dropdown','options'),
    [Input('cat_dropdown', 'value')],
    [State('cat_dropdown', 'value')],
)
def update_menu(val1, val2):
    pair_list['pair_name']=pair_list['ticker1_y']+'/'+pair_list['ticker2_y']
    select_pairs=pair_list[pair_list['type']==val1]
    pairs = select_pairs['pair_name'].copy()
    pairs.sort_values(inplace=True)
    print(pairs)
    pair_option = [{"label": i, "value": i} for i in pairs]

    return pair_option


@app.callback(
    Output('status_summ','children'),
    [Input('rv_data_button', 'n_clicks')],
    [State('rv_data_button', 'n_clicks'),
     State('pair_dropdown', 'value'),
     State('time_dropdown', 'value')],
)
def update_nclicks(n_clicks, n_clicks2, tickers, time_val):
    if n_clicks == 0:
        output = 'Select options from the dropdown'
    else: 
        if tickers == None or time_val == None:
            output = 'Select an RV pair to analyse'
        else:
            output = 'Getting data for '+tickers+'. Number of button clicks: '+str(n_clicks)
    return output
    

@app.callback(
    Output('rv_1','children'),
    [Input('rv_data_button', 'n_clicks')],
    [State('pair_dropdown', 'value'),
     State('time_dropdown', 'value')],
)
def update_rv_charts(n_clicks, tickers, time_val):
    if n_clicks == 0:
        raise PreventUpdate    
    if tickers == None or time_val == None:
        output = ''
    else:
        # Get basic info for the pairs
        ticker1 = pair_list[pair_list['pair_name']==tickers].iloc[0]['ticker1_y']
        ticker2 = pair_list[pair_list['pair_name']==tickers].iloc[0]['ticker2_y']
        fvr = pair_list[pair_list['pair_name']==tickers].iloc[0]['fair_value_ratio']
        rv_type = pair_list[pair_list['pair_name']==tickers].iloc[0]['type']
        curr1 = curr_table.loc[ticker1]['trade_currency']
        curr2 = curr_table.loc[ticker2]['trade_currency']
        tick_list = [ticker1, ticker2, curr1+'USD=X', curr2+'USD=X']
        print(tickers, rv_type, ticker1, ticker2, curr1+'USD=X', curr2+'USD=X', fvr)

        # Download the data
        data = yf.download(tickers=tick_list, period=time_val, interval='1d')
        
        # Keep only the 'Close' column
        data = data['Close']
       
        # Fix the date index including time
        data.index = data.index.date
        df_out=pd.DataFrame()
        for col in data.columns:
            df_out = pd.concat([df_out,data[col].dropna()],axis=1)
        df_out.sort_index(ascending=True, inplace=True)
        data=df_out

        # Fix currency column if any of the assets are in USD or GBp
        data['USDUSD=X']=1
        if curr1 =='GBp' or curr2 == 'GBp':
            data['GBpUSD=X'] = data['GBPUSD=X']/100
        
        # Calculate currency adjusted ratio in two different ways and drop any nan's
        if curr1 == curr2:
            data['curr_ratio']=1
        else:
            data['curr_ratio'] = data[(curr1+'USD=X')]/data[(curr2+'USD=X')]  
        
        
        # Calculate ticker in USD
        data['tick1_USD'] = data[ticker1] * data[(curr1+'USD=X')]
        data['tick2_USD'] = data[ticker2] * data[(curr2+'USD=X')]
        # Calculate the ratios
        data['ratio']= data['tick1_USD'] / data['tick2_USD']
        data['ratio2'] = data[ticker1] * data['curr_ratio'] / data[ticker2]
        data = data[data['ratio'].notna()]
        
        
        #Get the critical data
        ratio, summ = get_rv_stats_tearsheet(ticker1, ticker2, fvr, rv_type, data)
        
        
        
        
        output=[]
        
        # Historical spread distribution
        group_labels = ['distplot'] # name of the dataset
        hist_data=list(ratio['ratio'])
        
        fig = ff.create_distplot([hist_data], group_labels, show_hist=False, histnorm='percent')
        fw=go.FigureWidget(fig)
        new_trace=dict(
                           x=[hist_data[-1]],
                           y=['displot'],
                           mode='markers',
                           marker=dict(color='red', symbol='x', size=10),
                           xaxis='x',
                           yaxis='y2',
                           name='current')
        
        fw1=go.FigureWidget(data=[fw.data[k] for k in range(2)]+[new_trace],
                              layout=fw.layout)
        fw1.update_layout(title='Probability Distribution Function for all ratio data', title_x=0.5, height = 550)


        # Summary Table
        columns = [
            dict(id='index', name='Metric'),
            dict(id=tickers, name=ticker1+' / '+ticker2, type='numeric', format=Format(precision=3))]        
        cell_format = {'fontFamily': 'Nunito Sans', 'text-align': 'left', 'whiteSpace': 'normal', 'height': 'auto', 'padding':'5px'}
        style_header={'fontWeight': 'bold', 'backgroundColor': '#343a40', 'color': 'white'} 


        # Append summary table and historical spread distribution to 
        output.append(
            dbc.Row([dbc.Col(
                DataTable(data = summ.reset_index().to_dict('records'), 
                        columns=columns, 
                        style_cell = cell_format, 
                        style_header = style_header), width=3),
                
                dbc.Col(dcc.Graph(figure=fw1), width=9)
            ]))
        

        # Create chart for summary ratios  
        traces = []
        traces.append(go.Scatter(x=ratio.index, y=ratio['ratio'], mode='lines', name='ratio', line = {'color':'black','width':1}))
        for key in periods.keys():
            traces.append(go.Scatter(x=ratio.index, 
                                     y=ratio['{}_avg'.format(key)], 
                                     mode='lines',
                                     line={'width':1},
                                     name='{} avg'.format(key)))        
        
        chart_name = '{} rolling average ratios'.format(tickers)
        form=''
        
        output.append(
            dbc.Row(dbc.Col(
                dcc.Graph(
                figure={'data':traces,
                'layout': go.Layout(
                    title = {'text':chart_name, 'x':0.5},
                    yaxis = {'zeroline':True, 'zerolinecolor': 'black', 'tickformat':form},
                    height = 550,
                    )}
                ), width=12))
            )    

        # Stock price history in USD
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ratio.index, y=ratio['tick1_USD'], mode='lines', name=ticker1, yaxis='y1', line = {'width':1}))
        fig.add_trace(go.Scatter(x=ratio.index, y=ratio['tick2_USD'], mode='lines', name=ticker2, yaxis='y1', line = {'width':1}))     
        fig.update_layout(title='{} stock prices in USD'.format(tickers), title_x=0.5, height = 550,
                          yaxis_title='Stock Price USD') 

        output.append(
            dbc.Row(
                dbc.Col(dcc.Graph(figure=fig), width=12)
                ))



        # Stock price history chart (local currency)
        traces = []
        traces.append(go.Scatter(x=ratio.index, y=ratio[ticker1], mode='lines', name=ticker1, yaxis='y1', line = {'width':1}))
        traces.append(go.Scatter(x=ratio.index, y=ratio[ticker2], mode='lines', name=ticker2, yaxis='y2', line = {'width':1}))


        chart_name = '{} stock prices'.format(tickers)
        output.append(
            dbc.Row(dbc.Col(
                dcc.Graph(
                figure={'data':traces,
                'layout': go.Layout(
                    title = {'text':chart_name, 'x':0.5},
                    yaxis = {'title':ticker1+' price ('+curr1+')', 'zeroline':True, 'zerolinecolor': 'black', 'tickformat':form},
                    yaxis2={'title':ticker2+' price ('+curr2+')', 'overlaying':'y', 'side':'right'},                    
                    height = 550,
                    )}
                ), width=12))
            )    


        # Summary of correlations
        fig = go.Figure()
        # fig.add_trace(go.Scatter(x=ratio.index, y=ratio['ratio'], mode='lines', name='ratio', line = {'color':'black','width':1}))
        
        for key in periods.keys():
            fig.add_trace(go.Scatter(x=ratio.index, 
                                     y=ratio['{}_corr'.format(key)], 
                                     mode='lines',
                                     line={'width':1},
                                     name='{} correlation'.format(key)))
        
        fig.update_layout(title='{} rolling correlations'.format(tickers), title_x=0.5,
                            yaxis_title='Correlation',
                            yaxis={'tickformat':".0%"},
                            height=550)
        output.append(
            dbc.Row(
                dbc.Col(dcc.Graph(figure=fig), width=12)
                ))


        # Summary of all Z-scores
        fig=go.Figure()
        for key in periods:
            fig.add_trace(go.Scatter(x=ratio.index, y=ratio['{}_z-score'.format(key)], 
                                     mode='lines', name='{} z-score'.format(key), line = {'width':1}))        
        fig.update_layout(title='{} Z-scores'.format(tickers), title_x=0.5, yaxis_title='Z-Score', height=550)
        
        output.append(
            dbc.Row(
                dbc.Col(dcc.Graph(figure=fig), width=12)
                ))


        # Add all the z-score charts
        for key in list(periods.keys())[::-1]:         
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ratio.index, y=ratio['ratio'], mode='lines', name='ratio', line = {'color':'black','width':1}))
            fig.add_trace(go.Scatter(x=ratio.index, y=ratio['{}_avg'.format(key)], 
                                     mode='lines', name='{} avg'.format(key), line = {'color':'red'}))
            
            for i in ['-3','-2','-1','+1','+2','+3']:
                fig.add_trace(go.Scatter(x=ratio.index, y=ratio['{}_{}std'.format(key,i)], 
                                         mode='lines', name='{}std'.format(i), 
                                         line = {'color':'orangered', 'dash':'dash','width':1}))
            fig.update_layout(title='{} ratio {}'.format(tickers, key), title_x=0.5,
                               yaxis_title='Ratio', height=550)
            output.append(
                dbc.Row(
                    dbc.Col(dcc.Graph(figure=fig), width=12)
                    ))        


        # Historical standard deviations
        fig = go.Figure()
        for key in periods:
            fig.add_trace(go.Scatter(x=ratio.index, y=ratio['{}_std'.format(key)], 
                                     mode='lines', name='{} std_dev'.format(key), line = {'width':1}))
        fig.update_layout(title='{} ratio standard deviations'.format(tickers), title_x=0.5, height=550,
                           yaxis_title='standard deviation',
                           yaxis={'tickformat':".0%"})
        output.append(
            dbc.Row(
                dbc.Col(dcc.Graph(figure=fig), width=12)
                ))

    return output




# needed only if running this as a single page app
# =============================================================================
# if __name__ == '__main__':
#     app.run_server(debug=True)
# =============================================================================
