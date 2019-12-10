import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
import os

def trim_excel(df_to_trim, cut_points, set_sel):
    
    """
    trim_excel(df_to_trim, cut_points, set_sel)
    
    Accepts a data frame from an excel file with two data sets on one sheet (df_to_trim), and truncates the rows
    according to which data set is requested (set_sel) using a list of numerical index cut points (cut_points).
    """

    if set_sel == 1:
        trim_df = df_to_trim.drop([i for i in range(cut_points[0], cut_points[1])])
    elif set_sel == 2:
        trim_df = df_to_trim.drop([i for i in range(0, cut_points[1] - cut_points[0])])
    
    return trim_df



def first_item(df_to_split, index_to_split):
    
    """
    first_item(df_to_split, index_to_split)
    
    Accepts a data frame (df_to_split) with cells of type: str, and a specified index (index_to_split) for the data
    frame, and returns a list of the first item in each string.
    """
    
    keep_vals = []
    for ii in df_to_split.loc[index_to_split]:
        if pd.isna(ii):
            keep_vals.append(ii)
        else:
            holder = ii.split()
            keep_vals.append(holder[0])
    return keep_vals



def clean_excel(df_to_clean):
    
    """
    clean_excel(df_to_clean)
    
    Accepts a data frame (df_to_clean) from an Excel file with a predetermined format and populates multilevel index
    and column labels from cells within the dataframe, while removing the corresponding cells from the dataframe.
    """
    
    # Removes the index label columns from the input dataframe and cleans them to reattach later.
    index_df = df_to_clean.iloc[:, 0:6]
    index_df = index_df.dropna()
    index_df.columns = index_df.iloc[0]
    index_df = index_df.reset_index().drop(0).reset_index()
    
    # Removes the column label rows from the input dataframe and cleans them to reattach later.  Will need to add
    # items to repl_dic dictionary for any additional experimental conditions used.
    col_df = df_to_clean.iloc[0:3, 6:]
    repl_dic = {r'.*Aqueous.*':'Aq',
                r'^Day 28.*':'D28-De',
                r'^Day 6.*':'D6-De',
                r'^Day 14.*':'D14-De',
                r'.*Desiccated.*':'De',
                r'^Initial.*':'Init_Con'}
    col_df = col_df.replace(regex=repl_dic)
    col_df = col_df.fillna(method='ffill', axis=1)
    col_list = ['Condition', 'Dose', 'Dilution']
    col_df.index = col_list
    col_df.loc['Dose'] = first_item(col_df, 'Dose')
    col_end = len(col_df.columns) + 6
    col_tups = []
    for i in range(6, col_end):
        col_tups.append((col_df.loc['Condition', i],
                         col_df.loc['Dose', i],
                         col_df.loc['Dilution', i]))
    
    # Basic error check for values missing in the column header space of the input excel file.  If there is a
    # problem with the header, most likely outcome is a duplicated column header tuple (# unique < # total).
    if len(set(col_tups)) < len(col_tups):
        raise ValueError('Input Excel file column header formatting issue.  Please correct input file.')
    
    # Removes data values from the input dataframe and sets them to numbers (float or int).
    clean_df = df_to_clean.iloc[3:, 6:].apply(pd.to_numeric)
    
    # Sets clean_df index and column labels, then returns the result.
    clean_df.index = [index_df['Temp'], index_df['Triplicate']]
    clean_df.columns = pd.MultiIndex.from_tuples(col_tups, names = col_list)
    
    return clean_df



def value_removal(df_to_edit, r_list):
    
    """
    value_removal(df_to_edit, r_list)
    
    From a processed data frame (df_to_edit), removes a list of data values (r_list, type list of 3-element tuples) from the data
    that have been determined as unnecessary or erroneous.
    """
    
    # Extracts the location data from a tuple, then sets the corresponding data point as Nan.
    for tup_id in r_list:
        cond_val, temp_val, dose_val = tup_id
        df_to_edit.loc[cond_val, temp_val].loc[dose_val, 'mean'] = np.nan
    
    return df_to_edit