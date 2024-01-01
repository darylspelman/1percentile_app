from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import pandas as pd
import numpy as np

#Needed when running in full environment
from app import app



# LOAD AND CLEAN DATA
df = pd.read_pickle('data/business_quality_data.pkl')

# Eliminate entries with no data
df.dropna(subset=['mar_cap_mUSD','3m_val_turnover_mUSD', 'sector', 'BQ_score', 'V_score'], inplace=True)

# Clean sectors up
df = df[(df['sector'] != '') & (df['sector'] != 'Industrial Goods')]

# Eliminate the smallest items
df = df[df['3m_val_turnover_mUSD']>0]
df = df[df['mar_cap_mUSD']>0]



# PREP THE COMPONENTS OF THE CHART
# Define axes limits
overall_x_min = df['BQ_score'].min() * 1.05
overall_x_max = df['BQ_score'].max() * 1.05
overall_y_min = df['V_score'].min() * 1.05
overall_y_max = df['V_score'].max() * 1.05

# Logarithmic slider for mar_cap_mUSD
min_log_mar_cap = np.log10(df['mar_cap_mUSD'].min())
max_log_mar_cap = np.log10(df['mar_cap_mUSD'].max())

# Logarithmic slider for 3m_val_turnover_mUSD
min_log_val_turnover = np.log10(df['3m_val_turnover_mUSD'].min())
max_log_val_turnover = np.log10(df['3m_val_turnover_mUSD'].max())

# Sorted lists for dropdown menu
sorted_sector = sorted(df['sector'].unique())
sorted_country = sorted(df['country_name'].unique())



# The APP
# needed only if running this as a single page app
# =============================================================================
# app = dash.Dash(__name__)
# =============================================================================



## DEFINE THE LAYOUT
layout = html.Div([
        dbc.Container([
            dbc.Row(
                dbc.Col(html.H1("Compound Score Screen"
                        , className="text-center")
                        , className="mb-5 mt-5", style={'padding-top':'20px'})
            ),
            # Market cap slider
            html.Label("Market Cap (mUSD) Range"),
            dcc.RangeSlider(
                id='mar-cap-range-slider',
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
                id='val-turnover-slider',
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
                id='primary-checkbox',
                options=[{'label': 'Primary Only', 'value': 'Y'}],
                value=[]
            ),
            html.Br(),
            
            # Dropdown for selecting country
            html.Label("Select Country:"),
            dcc.Dropdown(
                id='country-dropdown',
                options=[{'label': 'All', 'value': 'All'}] +
                        [{'label': country, 'value': country} for country in sorted_country],
                value='All',
                multi=False
            ),
            html.Br(),            
            
            # Dropdown for selecting sector
            html.Label("Select Sector:"),
            dcc.Dropdown(
                id='sector-dropdown',
                options=[{'label': 'All', 'value': 'All'}] +
                        [{'label': sector, 'value': sector} for sector in sorted_sector],
                value='All',
                multi=False
            ),
            html.Br(),
            
            # Dropdown for selecting industry
            html.Label("Select Industry:"),
            dcc.Dropdown(
            id='industry-dropdown',
            value='All',
            multi=False
            ),
            html.Br(),
        
            # Outputs number of stocks selected
            html.Div(id='display-count', children="Number of items displayed: ", style={'font-weight': 'bold', 'color': 'blue'}),
            dcc.Graph(id='scatter-plot')                
        ]), #, style={'width': '48%', 'display': 'inline-block'}),
    ])



## Update industry pulldown when sector selected
@app.callback(
    Output('industry-dropdown', 'value'),
    [Input('sector-dropdown', 'value')]
)
def update_industry_dropdown(sector_value):
    return 'All'



## Update chart when inputs changed
@app.callback(
    [Output('scatter-plot', 'figure'),
     Output('display-count', 'children'),
     Output('industry-dropdown', 'options')],
    [
        Input('mar-cap-range-slider', 'value'),
        Input('val-turnover-slider', 'value'),
        Input('primary-checkbox', 'value'),
        Input('country-dropdown', 'value'),
        Input('sector-dropdown', 'value'),
        Input('industry-dropdown', 'value')
    ]
)

def update_figure(log_mar_cap_range, log_val_turnover_min, primary_check, selected_country, selected_sector, selected_industry):
    # Convert log values back to original values
    mar_cap_range = [10**log_value for log_value in log_mar_cap_range]
    val_turnover_min = 10**log_val_turnover_min
    
    # Apply the slider filter values
    filtered_df = df[(df['mar_cap_mUSD'] >= mar_cap_range[0]) & 
                     (df['mar_cap_mUSD'] <= mar_cap_range[1]) & 
                     (df['3m_val_turnover_mUSD'] >= val_turnover_min)]
    
    # If "Primary Only" is checked, filter dataframe based on the 'primary' column
    if 'Y' in primary_check:
        filtered_df = filtered_df[filtered_df['primary'] == 'Y']

    available_industries = [{'label': 'All', 'value': 'All'}]
    # Filtering by selected sector if not "All"
    if selected_sector != "All":
        filtered_df = filtered_df[filtered_df['sector'] == selected_sector]
        industries_for_selected_sector = sorted(df[df['sector'] == selected_sector]['industry'].unique())
        available_industries += [{'label': industry, 'value': industry} for industry in industries_for_selected_sector]
 
    # If an industry other than "All" is selected
    if selected_industry != 'All':
        filtered_df = filtered_df[filtered_df['industry'] == selected_industry]

    # If a country other than "All" is selected
    if selected_country != 'All':
        filtered_df = filtered_df[filtered_df['country_name'] == selected_country]


    # Define the legend structure and sort
    if selected_country == 'All':
        legend = 'country_name'
    elif selected_sector == 'All':
        legend = 'sector'
    else:
        legend = 'industry'
    sorted_legend = sorted(filtered_df[legend].unique())
    
    
    # Define each legend series to be plotted
    traces = [
        go.Scatter(
            x=filtered_df[filtered_df[legend] == legend_label]['BQ_score'],
            y=filtered_df[filtered_df[legend] == legend_label]['V_score'],
            mode='markers',
            marker=dict(
                size=10,
                opacity=0.5
            ),
            name=legend_label,
                   text=filtered_df[filtered_df[legend] == legend_label].apply(
                       lambda row: f"{row.name}<br>{row['name']}<br>Market Cap: {int(row['mar_cap_mUSD']):,} mUSD" + 
                       f"<br>ADV: {int(row['3m_val_turnover_mUSD']):,.1f} mUSD<br>Sector: {row['sector']}<br>" + 
                       f"Industry: {row['industry']}<br>BQ Percentile: {row['BQ_score_percentile']*100:,.1f}<br>" +
                       f"Valuation Percentile: {row['V_score_percentile']*100:,.1f}",
            axis=1
            ),
            hoverinfo='text'
        ) for legend_label in sorted_legend
    ]


    # Define the final chart figure
    figure = {
        'data': traces,
        'layout': go.Layout(
            title='Screen of Business Quality vs Valuation Scores',
            xaxis={'title': 'Business Quality Score', 'range': [overall_x_min, overall_x_max]},
            yaxis={'title': 'Valuation Score', 'range': [overall_y_min, overall_y_max]},
            hovermode='closest',
            height=800,
            # width=800
        )
    }
    
    
    # Count the number of items in the filtered dataframe
    count = len(filtered_df)
    
    return figure, f"Number of items displayed: {count}", available_industries

    
    
# needed only if running this as a single page app
# =============================================================================
# if __name__ == '__main__':
#     app.run_server(debug=True)
# =============================================================================    
