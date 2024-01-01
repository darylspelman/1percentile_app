import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import pandas as pd
import numpy as np

from dash import callback_context

#Needed when running in full environment
from app import app



#######################
## TO ADD
## scatter plot of total revenue, $b vs market cap
##
######################



########### LOAD AND CLEAN DATA
df = pd.read_pickle('data/business_quality_data.pkl')

# Eliminate entries with no data
df.dropna(subset=['mar_cap_mUSD','3m_val_turnover_mUSD', 'sector', 'BQ_score', 'V_score'], inplace=True)

# Clean sectors up
df = df[(df['sector'] != '') & (df['sector'] != 'Industrial Goods')]

# Eliminate the smallest items
df = df[df['3m_val_turnover_mUSD']>0]
df = df[df['mar_cap_mUSD']>0]

# Logarithmic slider for mar_cap_mUSD
min_log_mar_cap = np.log10(df['mar_cap_mUSD'].min())
max_log_mar_cap = np.log10(df['mar_cap_mUSD'].max())

# Logarithmic slider for 3m_val_turnover_mUSD
min_log_val_turnover = np.log10(df['3m_val_turnover_mUSD'].min())
max_log_val_turnover = np.log10(df['3m_val_turnover_mUSD'].max())

# Sorted lists for dropdown menu
sorted_sector = sorted(df['sector'].unique())
sorted_country = sorted(df['country_name'].unique())



def plot_dotplot(filtered_short_df, ticker, x_axis, y_axis, x_axis_label, y_axis_label):
    """
    Plots a dot plot of 2 axes
    """
    
    # Extract the ticker row (assuming 'ticker' is a global variable or passed into the function)
    ticker_row = filtered_short_df.loc[ticker]
     
    # Define each legend series to be plotted
    traces = [
         go.Scatter(
             x=filtered_short_df.drop(ticker)[x_axis],
             y=filtered_short_df.drop(ticker)[y_axis],
             mode='markers',
             marker=dict(
                 size=10,
                 opacity=0.5
             ),
             name='Comparables',
                    text=filtered_short_df.drop(ticker).apply(
                        lambda row: f"{row.name}<br>{row['name']}<br>Market Cap: {int(row['mar_cap_mUSD']):,} mUSD" + 
                        f"<br>ADV: {int(row['3m_val_turnover_mUSD']):,.1f} mUSD<br>Sector: {row['sector']}<br>" + 
                        f"Industry: {row['industry']}<br>BQ Percentile: {row['BQ_score_percentile']*100:,.1f}<br>" +
                        f"Valuation Percentile: {row['V_score_percentile']*100:,.1f}",
             axis=1
             ),
             hoverinfo='text'
         )
     ]
    
    # Add a separate trace for the ticker row
    ticker_trace = go.Scatter(
         x=[ticker_row[x_axis]],
         y=[ticker_row[y_axis]],
         mode='markers',
         marker=dict(
             size=12,  # Slightly larger marker size
             color='red',  # Distinct color
             # line=dict(
             #     color='black',
             #     width=2
             # )
         ),
         name=ticker,  # Name it as the ticker
         hoverinfo='text',
         text=f"{ticker_row.name}<br>{ticker_row['name']}<br>Market Cap: {int(ticker_row['mar_cap_mUSD']):,} mUSD" +
               f"<br>ADV: {int(ticker_row['3m_val_turnover_mUSD']):,.1f} mUSD<br>Sector: {ticker_row['sector']}<br>" +
               f"Industry: {ticker_row['industry']}<br>BQ Percentile: {ticker_row['BQ_score_percentile']*100:,.1f}<br>" +
               f"Valuation Percentile: {ticker_row['V_score_percentile']*100:,.1f}"
    )
    
    # Append the ticker trace to the traces list
    traces.append(ticker_trace)
    
    # Define the final chart figure
    figure = {
         'data': traces,
         'layout': go.Layout(
             title=f'{x_axis_label} vs {y_axis_label}',
             xaxis={'title': x_axis_label},
             yaxis={'title': y_axis_label},
             hovermode='closest',
             height=500,
             legend=dict(
                 x=0.05,  # Horizontal position (0 is far left)
                 y=0.95,  # Vertical position (1 is top)
                 xanchor='left',  # Anchor point for x
                 yanchor='top',  # Anchor point for y
                 bgcolor='rgba(255,255,255,0.8)',  # Semi-transparent white background
                 bordercolor='#DEE5ED',  # Border color
                 borderwidth=1  # Border width
             )
         )
     }
    return figure


def plot_bar_chart(df_pareto, ticker, column, title):
    """
    Returns:
    - A Plotly figure object containing the Pareto chart.
    """
    # Sort the DataFrame by the column
    sorted_df = df_pareto.sort_values(by=column, ascending=True)
    sorted_df.dropna(subset=column, inplace=True)

    # Create a list of colors, with a different color for the 'ticker'
    colors = ['#558BB8' if idx != ticker else 'red' for idx in sorted_df.index]

    # Generate hover text for each bar
    hover_text = [
        f"{idx}<br>{row['name']}<br>Market Cap: {int(row['mar_cap_mUSD']):,} mUSD" +
        f"<br>ADV: {int(row['3m_val_turnover_mUSD']):,.1f} mUSD<br>Sector: {row['sector']}<br>" +
        f"Industry: {row['industry']}<br>BQ Percentile: {row['BQ_score_percentile']*100:,.1f}<br>" +
        f"Valuation Percentile: {row['V_score_percentile']*100:,.1f}<br>" +
        f"{title}: {row[column]:,.1f}"
        for idx, row in sorted_df.iterrows()
    ]

    # Create the bar trace for the individual values
    bar_trace = go.Bar(
        x=sorted_df[column],
        y=sorted_df.index,
        orientation='h',
        marker=dict(color=colors),
        text=hover_text,  # Add hover text
        hoverinfo='text',
        textposition='none'  # Hide text labels on the bars
    )

    # Calculate the dynamic height
    height_per_item = 30  # Adjust this value based on your preference
    dynamic_height = min(len(sorted_df) * height_per_item + 150,750)
    dynamic_height = max(dynamic_height, 225)


    # Define the layout for the plot
    layout = go.Layout(
        title=title,
        height=dynamic_height  # Set the dynamic heigh
    )

    # Create and return the figure
    fig = go.Figure(data=[bar_trace], layout=layout)
    return fig




######## DEFINE THE LAYOUT
layout = html.Div([
        dbc.Container([
            dbc.Row(
                dbc.Col(html.H1("Comparable Company Metrics"
                        , className="text-center")
                        , className="mb-5 mt-5", style={'padding-top':'20px'})
            ),
            
        html.Div([
            dcc.Input(id='ticker-input-comps', type='text', style={'margin-right': '10px'}),  # Default value is AAPL
            html.Button(id='submit-button-comps', n_clicks=0, children='Submit', style={'margin-left': '10px'}),
        ], style={'padding': '10px'}),

        # Give ticker name and some text
        dbc.Row(dbc.Col(html.H2(id='ticker-message-comps'                     
                 , className="text-center")
                , className="mb-3 mt-3")),
        dbc.Row(dbc.Col(html.Div(id='stock-info-comps'                     
                 , className="text-left")
                , className="mb-3 mt-3")),
        dbc.Row(None, style={'margin-left': '10px', 'margin-top': '20px'}), # Add some vertical space between elements
            
            
            
            # Market cap slider
            html.Label("Market Cap (mUSD) Range"),
            dcc.RangeSlider(
                id='mar-cap-range-slider-comps',
                min=min_log_mar_cap,
                max=max_log_mar_cap,
                value=[min_log_mar_cap, max_log_mar_cap],
                marks={i: '{:,.0f}'.format(10**i) for i in range(int(min_log_mar_cap), int(max_log_mar_cap) + 1)},
                step=0.1
            ),
            html.Br(),
            
            # ADV slider
            html.Label("3m Average Value Traded (mUSD) Minimum"),
            dcc.Slider(
                id='val-turnover-slider-comps',
                min=min_log_val_turnover,
                max=max_log_val_turnover,
                value=min_log_val_turnover,
                marks={i: '{:,.0f}'.format(10**i) for i in range(int(min_log_mar_cap), int(max_log_mar_cap) + 1)},
                step=0.1
            ),
            html.Br(),
            
            # Primary stock selector
            html.Label("Primary Stock Only:"),
            dcc.Checklist(
                id='primary-checkbox-comps',
                options=[{'label': 'Primary Only', 'value': 'Y'}],
                value=['Y']
            ),
            html.Br(),
            
            # Dropdown for selecting country
            html.Label("Select Country:"),
            dcc.Dropdown(
                id='country-dropdown-comps',
                options=[{'label': 'All', 'value': 'All'}] +
                        [{'label': country, 'value': country} for country in sorted_country],
                value='All',
                multi=True
            ),
            html.Br(),            
            
            # Dropdown for selecting sector
            html.Label("Select Sector:"),
            dcc.Dropdown(
                id='sector-dropdown-comps',
                options=[{'label': 'All', 'value': 'All'}] +
                        [{'label': sector, 'value': sector} for sector in sorted_sector],
                value='All',
                multi=False
            ),
            html.Br(),
            
            # Dropdown for selecting industry
            html.Label("Select Industry:"),
            dcc.Dropdown(
            id='industry-dropdown-comps',
            value='All',
            multi=True
            ),
            html.Br(),
        
            # Outputs number of stocks selected
            html.Div(id='display-count-comps', children="Number of items displayed: ", style={'font-weight': 'bold', 'color': 'blue'}),
            dbc.Row(None, style={'margin-left': '10px', 'margin-top': '40px'}), # Add some vertical space between elements
            
            # Scatter plots
            dcc.Graph(id='scatter-plot-BQ'),
            dcc.Graph(id='scatter-plot-GM'),
            dcc.Graph(id='scatter-plot-ROE'),
            
            # Bar charts
            dcc.Graph(id='hist_GM'),
            dcc.Graph(id='hist_OPM'),
            dcc.Graph(id='hist_SGA'),
            dcc.Graph(id='hist_ROE'),
            dcc.Graph(id='hist_RNOA'),
            dcc.Graph(id='hist_growth-1yr'),
            dcc.Graph(id='hist_growth'),
            
            dcc.Graph(id='hist_leverage'),
            dcc.Graph(id='hist_R&D'),
            dcc.Graph(id='hist_S&M'),
            dcc.Graph(id='hist_capex'),
            dcc.Graph(id='hist_reinvest'),
            dcc.Graph(id='hist_PEttm'),
        ]), #, style={'width': '48%', 'display': 'inline-block'}),
    ])


# Set the default value at the start for the Input field
@app.callback(
    Output('ticker-input-comps', 'value'),
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
    Input('submit-button-comps', 'n_clicks'),
    dash.dependencies.State('ticker-input-comps', 'value'),
    prevent_initial_call=True
)
def update_shared_data(n_clicks, ticker_value):
    if n_clicks is not None and ticker_value is not None:
        return {'ticker': ticker_value.upper().strip()}
    return dash.no_update



# Callback for ticker message
@app.callback(
    [Output('ticker-message-comps', 'children'),
     Output('stock-info-comps', 'children'),
     Output('country-dropdown-comps', 'value'),
     Output('sector-dropdown-comps', 'value'),
     Output('industry-dropdown-comps', 'value')],  # Two outputs: one for the plot and one for the message
    [Input('submit-button-comps', 'n_clicks'),
     Input('shared-data', 'data')],
    [dash.dependencies.State('ticker-input-comps', 'value')]
)
def update_figure_and_message(n_clicks, stored_ticker, ticker):
    if ticker == None:
        ticker = stored_ticker['ticker']
    ticker = ticker.upper().strip()

    if ticker not in df.index:
        return go.Figure(), 'Ticker not in database', None, 'All'  # No update to the graph, message updated
    else:
        name = df.loc[ticker]['name'] + ' (' + ticker + ')'
        info = [f"Market Cap (bUSD):  {df.loc[ticker]['mar_cap_mUSD']/1000:,.1f}",
                html.Br(),
                f"ADV (mUSD): {df.loc[ticker]['3m_val_turnover_mUSD']:,.1f}",
                html.Br(),
                f"Sector: {df.loc[ticker]['sector']}",
                html.Br(),
                f"Industry: {df.loc[ticker]['industry']}"]
        country_value = df.loc[ticker]['country_name']
        sector_value = df.loc[ticker]['sector']
        industry_value = df.loc[ticker]['industry']
        return name, info, country_value, sector_value, industry_value  # Graph updated, message updated with name


## Update chart when inputs changed
@app.callback(
    [Output('display-count-comps', 'children'),
     Output('industry-dropdown-comps', 'options'),
     
     Output('scatter-plot-BQ', 'figure'),
     Output('scatter-plot-GM', 'figure'),
     Output('scatter-plot-ROE', 'figure'),
     
     Output('hist_GM', 'figure'),
     Output('hist_OPM', 'figure'),
     Output('hist_SGA', 'figure'),
     Output('hist_ROE', 'figure'),
     Output('hist_RNOA', 'figure'),
     Output('hist_growth-1yr', 'figure'),
     Output('hist_growth', 'figure'),
     Output('hist_leverage', 'figure'),
     Output('hist_R&D', 'figure'),
     Output('hist_S&M', 'figure'),
     Output('hist_capex', 'figure'),
     Output('hist_reinvest', 'figure'),
     Output('hist_PEttm', 'figure'),
     ],
    [
        Input('mar-cap-range-slider-comps', 'value'),
        Input('val-turnover-slider-comps', 'value'),
        Input('primary-checkbox-comps', 'value'),
        Input('country-dropdown-comps', 'value'),
        Input('sector-dropdown-comps', 'value'),
        Input('industry-dropdown-comps', 'value'),
        Input('shared-data', 'data')],
    [dash.dependencies.State('ticker-input-comps', 'value')]
)

def update_figure(log_mar_cap_range, log_val_turnover_min, primary_check, selected_country, selected_sector, selected_industry, 
                  stored_ticker, ticker):

    if ticker == None:
        ticker = stored_ticker['ticker']
    ticker = ticker.upper().strip()    
    print(ticker)
    
    ## PREP THE INPUT DATA
    # Convert log values back to original values for range sliders
    mar_cap_range = [10**log_value for log_value in log_mar_cap_range]
    val_turnover_min = 10**log_val_turnover_min
    
    # Convert ticker from a list to a string
    # ticker = ticker[0]
    
    # Convert selected country to a list if not already. First instance is always a string and then subsequently always a list
    if isinstance(selected_country, str):
        selected_country = [selected_country]
    if isinstance(selected_industry, str):
        selected_industry = [selected_industry]
    
    
    ## CREATE THE AVAILABLE INDUSTRY OUTPUT LIST
    available_industries = [{'label': 'All', 'value': 'All'}]
    # Filtering by selected sector if not "All"
    if selected_sector != "All":
         industries_for_selected_sector = sorted(df[df['sector'] == selected_sector]['industry'].unique())
         available_industries += [{'label': industry, 'value': industry} for industry in industries_for_selected_sector]    
    
    
    ## CREATE FILTERED DATA FRAME
    # Apply the slider filter values
    filtered_df = df[(df['mar_cap_mUSD'] >= mar_cap_range[0]-1) & 
                     (df['mar_cap_mUSD'] <= mar_cap_range[1]+1) & 
                     (df['3m_val_turnover_mUSD'] >= val_turnover_min)]
    
    # If "Primary Only" is checked, filter dataframe based on the 'primary' column
    if 'Y' in primary_check:
        filtered_df = filtered_df[filtered_df['primary'] == 'Y']

    # print(selected_country, ";", selected_sector, ";", selected_industry)

    # Filter by country if a country other than "All" is selected
    if 'All' not in selected_country and len(selected_country)>0:
        filtered_df = filtered_df[filtered_df['country_name'].isin(selected_country)]
 
    # Filter by sectors
    if selected_sector != "All":
        filtered_df = filtered_df[filtered_df['sector'] == selected_sector]

        # If an industry other than "All" is selected
        if 'All' not in selected_industry and len(selected_industry)>0:
            industry_list = [industry['value'] for industry in available_industries]
            if selected_industry[0] in industry_list: # f the selected_industry list is still relevant then use it to filter
                filtered_df = filtered_df[filtered_df['industry'].isin(selected_industry)]
    
    # Add the ticker row back in if data is missing
    if ticker not in filtered_df.index:
        ticker_row_df = df.loc[[ticker]]
        filtered_df = pd.concat([filtered_df, ticker_row_df])

    # Sort and trim the dataframe    
    filtered_df.sort_values(by='mar_cap_mUSD', inplace=True, ascending=False)
    count_before = len(filtered_df)
    filtered_short_df = filtered_df.iloc[0:100]
    count_after = len(filtered_short_df)
    return_text = [f"Total number of comparable companies found: {count_before}",
                   html.Br(),
                   f"Total number of comparable companies displayed: {count_after}"]
    

    figure_bq_vs = plot_dotplot(filtered_short_df, ticker, 'BQ_score', 'V_score', 'Business Quality Score', 'Valuation Score')
    figure_gm_rev = plot_dotplot(filtered_short_df, ticker, 'Revenue 3-yr CAGR', 'GPM%', 'Revenue Growth 3-year CAGR', 'Gross Profit Margin')
    figure_roe_pb = plot_dotplot(filtered_short_df, ticker, 'ROE', 'P/B', 'Return on Equity', 'Price / Book')
    
    
    fig_gm_hist = plot_bar_chart(filtered_short_df, ticker, 'GPM%', 'Gross Profit Margins')
    fig_opm_hist = plot_bar_chart(filtered_short_df, ticker, 'OPMargin', 'Operating Profit Margins')
    fig_sga_hist = plot_bar_chart(filtered_short_df, ticker, 'SGA_pctSales', 'SG&A / Sales')
    fig_roe_hist = plot_bar_chart(filtered_short_df, ticker, 'ROE', 'Return on Equity')
    fig_rnoa_hist = plot_bar_chart(filtered_short_df, ticker, 'RNOA', 'Return on Net Operating Assets')    
    fig_growth_1yr_hist = plot_bar_chart(filtered_short_df, ticker, 'Revenue 1-yr CAGR', 'Revenue Growth (1-yr)')
    fig_growth_hist = plot_bar_chart(filtered_short_df, ticker, 'Revenue 3-yr CAGR', 'Revenue Growth (3-yr CAGR)')
    fig_leverage_hist = plot_bar_chart(filtered_short_df, ticker, 'assets/equity', 'Assets / Leverage')
    fig_randd_hist = plot_bar_chart(filtered_short_df, ticker, 'R&D_pctSales', 'R&D / Sales')
    fig_sandm_hist = plot_bar_chart(filtered_short_df, ticker, 'S&M%sales', 'S&M / Sales')
    fig_capex_hist = plot_bar_chart(filtered_short_df, ticker, 'capitalExpenditures_pctSales', 'Capex / Sales')
    fig_invest_hist = plot_bar_chart(filtered_short_df, ticker, 'total_investment/sales', 'Total Investments / Sales')
    fig_pe_hist = plot_bar_chart(filtered_short_df, ticker, 'P/E_ttm', 'Trailing Price Earnings Ratio')
    
    return (return_text, available_industries, 
            figure_bq_vs, figure_gm_rev, figure_roe_pb, 
            fig_gm_hist, fig_opm_hist, fig_sga_hist, fig_roe_hist, fig_rnoa_hist, fig_growth_1yr_hist, fig_growth_hist, fig_leverage_hist,
            fig_randd_hist, fig_sandm_hist, fig_capex_hist, fig_invest_hist, fig_pe_hist)

    
    
# needed only if running this as a single page app
# =============================================================================
# if __name__ == '__main__':
#     app.run_server(debug=True)
# =============================================================================    
