#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 20:49:28 2021

@author: darylspelman
"""

import pandas as pd
import numpy as np
import math
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap


def format_pct(val):
    """
    Takes a percent number formatted in decimial point form and returns it as a string with % at end and 1 decimal point for use in outputs
    """
    
    if math.isnan(val) == True:
        val2=''
    else:
        val2 = str(round(val*100, 1))+'%'
    return val2


def normalize(x,a,b): #maps  the interval [a,b]  to [0,1]
    if a>=b:
        raise ValueError('(a,b) is not an interval')
    return float(x-a)/(b-a)


def asymmetric_colorscale(data,  div_cmap, ref_point=0.0, step=0.05):
    #data: data can be a DataFrame, list of equal length lists, np.array, np.ma.array
    #div_cmap is the symmetric diverging matplotlib or custom colormap
    #ref_point:  reference point
    #step:  is step size for t in [0,1] to evaluate the colormap at t
   
    if isinstance(data, pd.DataFrame):
        D = data.values
    elif isinstance(data, np.ma.core.MaskedArray):
        D=np.ma.copy(data)
    else:    
        D=np.asarray(data, dtype=np.float) 
    
    dmin=np.nanmin(D)
    dmax=np.nanmax(D)
    if not (dmin < ref_point < dmax):
        raise ValueError('data are not appropriate for a diverging colormap')
        
    if dmax+dmin > 2.0*ref_point:
        left=2*ref_point-dmax
        right=dmax
        
        s=normalize(dmin, left,right)
        refp_norm=normalize(ref_point, left, right)# normalize reference point
        
        T=np.arange(refp_norm, s, -step).tolist()+[s]
        T=T[::-1]+np.arange(refp_norm+step, 1, step).tolist()
        
        
    else: 
        left=dmin
        right=2*ref_point-dmin
        
        s=normalize(dmax, left,right) 
        refp_norm=normalize(ref_point, left, right)
        
        T=np.arange(refp_norm, 0, -step).tolist()+[0]
        T=T[::-1]+np.arange(refp_norm+step, s, step).tolist()+[s]
        
    L=len(T)
    T_norm=[normalize(T[k],T[0],T[-1]) for k in range(L)] #normalize T values  
    return [[T_norm[k], colors.rgb2hex(div_cmap(T[k]))] for k in range(L)]

    
def colormap_to_colorscale(cmap):
    #function that transforms a matplotlib colormap to a Plotly colorscale
    return [ [k*0.1, colors.rgb2hex(cmap(k*0.1))] for k in range(11)]    


def colorscale_from_list(alist, name): 
    # Defines a colormap, and the corresponding Plotly colorscale from the list alist
    # alist=the list of basic colors
    # name is the name of the corresponding matplotlib colormap
    
    cmap = LinearSegmentedColormap.from_list(name, alist)
    colorscale=colormap_to_colorscale(cmap)
    return cmap, colorscale


def heatmap_data(daily_price):
    """
    Takes in a pandas dataseries of daily spot prices.
    Returns a 2x2 by year and month of 
    """

    month_price = daily_price.groupby([daily_price.index.year,daily_price.index.month]).last()

    # Drop first row
    month_price = month_price.iloc[1:]
    month_pct=month_price.pct_change()

    # Create 2x2 layout from data series - create empty dataframe
    twodf = pd.DataFrame(columns=[1,2,3,4,5,6,7,8,9,10,11,12], index=month_pct.index.unique(level=0))

    # fill the dataframe from the data series
    for i in range(len(month_pct.index)):
        year = month_pct.index[i][0]
        month = month_pct.index[i][1]
        val = month_pct.iloc[i]
        twodf.loc[year][month] = val

    # Create labels for chart
    labeldf = twodf.copy()
    for col in labeldf.columns:
        labeldf[col] = labeldf[col].apply(format_pct)
        
    return twodf, labeldf