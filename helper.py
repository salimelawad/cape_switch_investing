#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 10:24:48 2019

@author: elawad
"""

import os
import pandas as pd
import numpy as np
import functools
import math
from functools import reduce
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

def cape_return_mapping(yearly_cape, year_percentage_returns):
    def extract_cape(country):
        frame = {'cape': yearly_cape[country], 'return': year_percentage_returns[country]}
        return pd.DataFrame(frame).dropna() 
    
    cape_year = reduce(
        lambda x, y: x + list(y.itertuples(index=False, name=None)),
        (map(
            lambda country: extract_cape(country)
            , list(yearly_cape.columns)
        )),
        []
    )

    cape_year = pd.DataFrame(cape_year, columns=['cape', 'return'])
    cape_year['return'] = cape_year['return']
    corr = cape_year.corr(method='spearman').loc['cape']['return']
    return cape_year, corr

def multi_year_returns(yearly_returns, years, adjust_yearly=False, adjust_inflation=False, inflation=None): 
    yearly_returns = yearly_returns.copy(deep=True)
    if adjust_inflation:
        yearly_returns['Inflation'] = inflation + 1
    r = yearly_returns.rolling(years, min_periods=years).aggregate(lambda x: x.prod()).shift(years*(-1)+1).dropna(how='all')
    if adjust_yearly:
        adjust_year_func = lambda x: math.exp(math.log(x, math.exp(1))/years)
        r = r.applymap(adjust_year_func)
    if adjust_inflation:
        r = r.apply(lambda x: x/r['Inflation']).drop(['Inflation'], axis=1)
    return r

def plot_cape_returns(data):
    sns.set(color_codes=True)
    fig, ax = plt.subplots(figsize=(25,10)) 
    ax.set_title('Relationship between country\'s CAPE ratios and 1-year returns', size= 30)
    ax = sns.regplot(x="cape", y="return", data=data, logx=True)
    plt.xlabel('CAPE ratio', size=20)
    plt.ylabel('Returns', size=20)

def ingest_cape_data():
    # Historical CAPE ratios

    historical_cape = pd.read_csv('historical_cape.csv', index_col='Date', parse_dates=['Date'])

    # Value of 0 indicates missing data, replace with with NaN. 
    historical_cape = historical_cape.replace(0, np.NaN)

    #Use first value of year for yearly value. This will be index value of last day in January for old indeces, 
    # or a value from a random month for new indeces. 
    year_cape = historical_cape.groupby(historical_cape.index.year).first()

    return year_cape

def _ingest_country_returns():
    # Read and Join MSCI yearly data

    countries = os.listdir("csv_cape_ratio_countries")

    country_data = [pd.read_csv("csv_cape_ratio_countries/"+country, index_col='Date', parse_dates=['Date'], thousands=',') for country in countries]

    all_returns = functools.reduce(lambda x, y: x.join(y, how='outer'), country_data)

    #Use first value of year for yearly value. This will be index value of last day in January for old indeces, 
    # or a value from a random month for new indeces. 
    all_returns_year = all_returns.groupby(all_returns.index.year).first()

    return all_returns_year

def compute_yearly_percentage(bonds):
    # Compute yearly percentage change. 
    # Value in year n should represent returns from January of year n to January of year n+1

    all_returns_year = _ingest_country_returns()

    all_returns_year_pct = all_returns_year.pct_change() 

    all_returns_year_pct = all_returns_year_pct + 1

    all_returns_year_pct = all_returns_year_pct.shift(periods=-1)
    
    all_returns_year_pct['Bonds'] = bonds

    return all_returns_year_pct

def ingest_asset_returns():

    assets_returns = pd.read_csv("asset_returns.csv", index_col='Year').apply(lambda s: s.str.rstrip('%').astype('float') / 100.0, axis=1)

    bond_returns = pd.concat([assets_returns['10-year Treasury'].loc[1972:1986],
                              assets_returns['Total US Bond Market'].loc[1987:1993],
                              assets_returns['Global Bonds (Unhedged)'].loc[1994:2019]])

    assets_returns['Bonds'] = bond_returns + 1

    return assets_returns



def get_yearly_investments(start_year, end_year, yearly_cape_data, methodology, cape_limit=None, count_limit=None, min_investments=0):
    if methodology == 'smallest':
        investments_per_year =  yearly_cape_data.apply(lambda s: s.nsmallest(count_limit).index.tolist(), axis=1).loc[start_year:end_year].to_dict()
    elif methodology == 'largest':
        investments_per_year = yearly_cape_data.apply(lambda s: s.nlargest(count_limit).index.tolist(), axis=1).loc[start_year:end_year].to_dict()
    elif methodology == 'lessthan':
        investments_per_year = yearly_cape_data.apply(lambda s: s[s.le(cape_limit)].index.tolist(), axis=1).loc[start_year:end_year].to_dict()
    elif methodology == 'greaterthan':
        investments_per_year = yearly_cape_data.apply(lambda s: s[s.ge(cape_limit)].index.tolist(), axis=1).loc[start_year:end_year].to_dict()
    elif methodology == 'smallest_limit':
        smallest = yearly_cape_data.apply(lambda s: s.nsmallest(count_limit).index.tolist(), axis=1).loc[start_year:end_year].to_dict()
        less_than = yearly_cape_data.apply(lambda s: s[s.le(cape_limit)].index.tolist(), axis=1).loc[start_year:end_year].to_dict()
        investments_per_year = {k: list(set(smallest[k]).intersection(set(less_than[k]))) for k, v in smallest.items()}
    elif methodology == 'all':
        investments_per_year = yearly_cape_data.apply(lambda s: s.nsmallest(1000).index.tolist(), axis=1).loc[start_year:end_year].to_dict()
    else:
        raise ValueError('methodology not implemented')

    investments_per_year = { k:v + ['Bonds'] * (min_investments - len(v)) for k,v in investments_per_year.items()}

    return investments_per_year

def calculate_investment_returns(yearly_investments, year_percentage_returns):
    yearly_returns = {}
    
    for year in yearly_investments.keys():
        returns_for_year = year_percentage_returns[yearly_investments[year]].loc[year].mean()
        if math.isnan(returns_for_year):
            returns_for_year = 1
        yearly_returns[year] = returns_for_year
        
    investment_returns = functools.reduce(lambda x, y: x*y, yearly_returns.values())

    return investment_returns, yearly_returns

def get_investment_returns(yearly_cape, year_percentage_returns, asset_returns, start_year, end_year, methodology, cape_limit=None, count_limit=None, min_investments=0, adjust_for_inflation=False):
    investments_per_year = get_yearly_investments(start_year, end_year, yearly_cape, methodology, cape_limit, count_limit, min_investments)
        
    investment_returns, yearly_returns = calculate_investment_returns(investments_per_year, year_percentage_returns)

    if adjust_for_inflation:
        investment_returns = investment_returns / asset_returns['Inflation'].loc[start_year:end_year].add(1).product()
   
    return(investment_returns, investments_per_year, yearly_returns)


if __name__ == '__main__':
    asset_returns = ingest_asset_returns()
    yearly_cape = ingest_cape_data()
    year_percentage_returns = compute_yearly_percentage(asset_returns['Bonds'])

#
#    print(get_investment_returns(yearly_cape, year_percentage_returns, asset_returns, 1982, 2018, 'all', adjust_for_inflation=True))
#    print(get_investment_returns(yearly_cape, year_percentage_returns, asset_returns, 1982, 2018, 'smallest', count_limit=100, adjust_for_inflation=True))
#    print(get_investment_returns(yearly_cape, year_percentage_returns, asset_returns, 1981, 2018, 'largest', count_limit=1, adjust_for_inflation=True))
#    print(get_investment_returns(yearly_cape, year_percentage_returns, asset_returns, 1981, 2018, 'lessthan', cape_limit=15, adjust_for_inflation=True))
#    print(get_investment_returns(yearly_cape, year_percentage_returns, asset_returns, 1981, 2018, 'greaterthan', cape_limit=30, adjust_for_inflation=True))
#    print(get_investment_returns(yearly_cape, year_percentage_returns, asset_returns, 1981, 2018, 'smallest_limit', count_limit = 5, cape_limit=16, min_investments=5, adjust_for_inflation=True))
#
    
    aFunc = lambda x, y: get_investment_returns(yearly_cape, year_percentage_returns, asset_returns, 1982, 2018, 'smallest_limit', count_limit = x+5, cape_limit=y+5, min_investments=5, adjust_for_inflation=True)[0]
    
    import matplotlib.pyplot as plt
    a = np.fromfunction(aFunc, (10, 10), dtype=int)
    plt.imshow(a, cmap='hot', interpolation='nearest')
    plt.show()
