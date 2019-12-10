import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
import os

def zero_normalize(df_to_norm):

    """
    zero_normalize(df_to_norm)
    
    Averages the 0 kGy dose readings for each condition and triplicate for an input data frame (df_to_norm), then
    normalizes the other dose readings by the zero average.
    """
    
    zero_normed_df = pd.DataFrame()
    for idx, condi in df_to_norm.groupby(level='Condition'):
        condi = condi.apply(lambda x: x / (condi['0'].mean(axis=1)))
        zero_normed_df = zero_normed_df.append(condi)
    return zero_normed_df



def data_calc(df_to_calc):
    
    """
    data_calc(df_to_calc)
    
    Receives formatted dataframe (df_to_calc), converts all values to log10, then calculates then mean and standard
    deviation for each 'Dose' in each ('Temp', 'Condition') combination.  The resulting dataframe is then rearranged
    for future plotting and packaging.
    """
    # Log10 conversion of data.
    log_df = np.log10(df_to_calc)
    
    # Aggregate data across 'Triplicate' and 'Dilution' to get averages and standard deviations for each dose.
    calced_df = log_df.stack().groupby(level=['Temp', 'Condition']).agg(['mean', 'std'])

    # Reorganize output dataframe for further processing.
    calced_df = calced_df.stack().reorder_levels([1, 0, 2])
    calced_df.columns = calced_df.columns.astype(float)
    calced_df = calced_df.unstack([1, 2]).stack(0).sort_index().sort_index(axis=1)
    
    return calced_df



def day_6_conv(zero_df):
    
    """
    day_6_conv(zero_df)
    
    From a processed data frame (zero_df) with two data sets taken under similar conditions, finds the highest common dose/value
    between the sets, generates a conversion factor, and modifies the latter data set to append its unique values to the former.
    """
    
    # Extracts the two similar data set indexes from the data frame for comparison.
    d6_index = zero_df.loc['D6-De', 'RT'].index
    d5_index = zero_df.loc['De', 'RT'].index

    # Identifies the highest common dose/value between the two data sets, and the data sets unique to the latter data set.
    common_value = d6_index.intersection(d5_index).max()
    append_values = [x for x in d6_index if x > common_value]
    
    # For both RT and -80 (F) data in the similar data sets, identifies the data values associated with the common_value dose.
    d6_RT_c_val = zero_df.loc['D6-De', 'RT'].loc[common_value, 'mean']
    d5_RT_c_val = zero_df.loc['De', 'RT'].loc[common_value, 'mean']

    d6_F_c_val = zero_df.loc['D6-De', -80].loc[common_value, 'mean']
    d5_F_c_val = zero_df.loc['De', -80].loc[common_value, 'mean']

    # From the above data values, calculates the conversion factors.
    des_RT_conv = d5_RT_c_val - d6_RT_c_val
    des_F_conv = d5_F_c_val - d6_F_c_val

    # Extracts the unique values from the latter data set and converts them using the above conversion factors.
    conv_df = zero_df.loc['D6-De'].loc[append_values[0]:]
    conv_df[('RT', 'mean')] = conv_df[('RT', 'mean')] + des_RT_conv
    conv_df[(-80, 'mean')] = conv_df[(-80, 'mean')] + des_F_conv

    # Appends the converted values above to the former data set.
    for ii in append_values:
        zero_df.loc[('De', ii), :] = conv_df.loc[ii]