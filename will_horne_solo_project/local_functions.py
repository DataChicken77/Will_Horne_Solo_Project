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



def csv_export(df_to_export, norm_term, strain_info):
    
    """
    csv_export(df_to_export, norm_term, strain_info)
    
    From an input processed data frame (df_to_export), generates .csv files for each ('Condition', 'Temp') data set.
    norm_term is either 'init' or 'zero' depending on the type of data frame passed, and is combined with strain_info
    to make appropriately formatted and descriptive file names. These files can be used for archival purposes and
    further statistical analysis outside of the original script.
    
    The files will be stored in './archive', relative to the current working directory.  If '/archive' does not exist,
    the function will create it.
    """
    
    # Makes a list of unique and sorted 'Condition' values to use for the iteration processes.
    i_term = 0
    cond_list = df_to_export.index.get_level_values('Condition')
    cond_list = sorted(list(set(cond_list)))
    
    # Checks for presence of './archive' directory, and creates it if it doesn't exist.
    if not os.path.isdir('./archive'):
        os.mkdir('archive')

    # Generate files for all condition sets with 'Temp' = -80.
    for idx, condi in df_to_export[-80].groupby('Condition'):
        csv_f_name = './archive/' + strain_info + '_' + cond_list[i_term] + '_' + '-80_' + norm_term + '.csv'
        condi.to_csv(path_or_buf=csv_f_name, index_label=['Condition', 'Dose'])
        i_term += 1

    # Generate files for all condition sets with 'Temp' = 'RT'.
    i_term = 0
    for idx, condi in df_to_export['RT'].groupby('Condition'):
        csv_f_name = './archive/' + strain_info + '_' + cond_list[i_term] + '_' + 'RT_' + norm_term + '.csv'
        condi.to_csv(path_or_buf=csv_f_name, index_label=['Condition', 'Dose'])
        i_term += 1



def request_file():
    
    """
    Brute force method to request a file from the user.  Currently set for only Excel file types.  Using this method
    to ensure file dialog box appears on top of all other windows.
    """
    
    from tkinter import Tk
    from tkinter.filedialog import askopenfilename
    
    # Make a top-level instance and hide from user.
    root = Tk()
    root.withdraw()

    # Make it almost invisible - no decorations, 0 size, top left corner.
    root.overrideredirect(True)
    root.geometry('0x0+0+0')

    # Show window again and lift it to top so it can get focus, otherwise dialogs will end up behind the terminal.
    root.deiconify()
    root.lift()
    root.focus_force()

    # Show an "Open" dialog box and return the path to the selected file
    file_path = askopenfilename(initialdir='./IR_Datasets/',
                                title='Excel to Read',
                                filetypes=(('New Excel', '*xlsx'), ('Old Excel', '*.xls')),
                                parent=root)

    # Get rid of the top-level instance once to make it actually invisible.
    root.destroy()
    
    return file_path


                
def data_display(df_to_graph, g_title, g_list, use_init=False):
    
    """
    data_display(df_to_graph, g_title, g_list, use_init=False)
    
    Determines if linear or quadratic regression best fits the specified data from a data frame, then plots the data and
    regression curve and saves a .png of the graph.
    
    Arguments:
    
    df_to_graph (type DataFrame):  Data frame output from analyze_cfu_counts(); holds data that will be graphed
    
    g_title (type str):  Title of the graph to be generated
    
    g_list (type list of lists):  List of the condition sets, stored as two-item lists themselves, to be graphed from the
    data
    
    use_init (type bool):  Determines if the data frame used is of initial- or zero-normalized data; as initial-normalized
    data should only be graphed to show differences in survival caused by the condition set and not the radiation, setting
    this value to True will cause the graph to focus on the 0 kGy values
    
    Returns:
    
    This function will plt.show() the resulting graph.  It will also save a .png file of the graph to
    './graphs/(g_title).png', relative to the current working directory.  The function will also create '/graphs' if it
    doesn't exist in the current directory.
    """
    
    # Modules needed to perform the regression analysis.
    import statsmodels.api as sm
    from sklearn.preprocessing import PolynomialFeatures
    
    # Establish the size and data markers for the graph.
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    marker_list = ['.', 's', '^', 'x']
    color_list = ['r', 'b', 'g', 'y']
    line_list = ['--', '-.', ':', '--']
    
    # Variables that will help establish the proper sequencing of labels in the plot legend.
    if len(g_list) == 2:
        leg_order = [2, 0, 3, 1]
    elif len(g_list) == 4:
        leg_order = [4, 0, 5, 1, 6, 2, 7, 3]
    
    # Sets the labels of the graph, and adjusts the graph range if focusing on the 0 kGy doses for an init data set.
    plt.title(g_title)
    plt.xlabel('Dose (kGy)')
    plt.ylabel('Log Survival')
    if use_init:
        plt.xlim(-0.1,3)
        plt.ylim(-2,1)
    
    # Loop that plots each individual condition set within g_list.
    for ii in range(0, len(g_list)):
        
        # Pulls the necessary x, y, and error values, as well as a label for the data, from the data frame.
        g_df = df_to_graph.loc[g_list[ii][0], g_list[ii][1]].dropna()
        x_vals = g_df.index.values
        y_vals = g_df['mean']
        yerr_vals = g_df['std']
        lin_label = str(g_list[ii][0]) + ' ' + str(g_list[ii][1])
        
        # Creates an x^2 data set for quadratic regression analysis.
        polynomial_features= PolynomialFeatures(degree=2)
        x_vals_p = polynomial_features.fit_transform(x_vals.reshape(-1,1))

        # Performs both linear and quadratic regression for the data set.
        lin_mod = sm.OLS(y_vals, x_vals).fit()
        quad_mod = sm.OLS(y_vals, x_vals_p).fit()
    
        # Compares the R^2 values for linear and quadratic, and checks if the quadratic term is positive (would result in an upward
        # survival curve at extreme radiation doses, which is a biological impossibilty).  Uses these conditions to determine if
        # linear or quadratic data will be retained and graphed.
        if (lin_mod.rsquared > quad_mod.rsquared) | (quad_mod.params[2] > 0):
            fin_mod = lin_mod
            fin_pred_x_vals = x_vals
        else:
            fin_mod = quad_mod
            fin_pred_x_vals = x_vals_p
    
        # Sets the y values and extracts the R^2 values for the selected regression method.
        ypred = fin_mod.predict(fin_pred_x_vals)
        r2_str = "%.3f" % fin_mod.rsquared
        r2_label = 'R^2 = ' + r2_str
        
        # Plot data points and regression curve, and print R^2 value.
        plt.errorbar(x_vals, y_vals, fmt='none', yerr=yerr_vals, ecolor=color_list[ii], capsize=3, alpha=0.7)
        plt.scatter(x_vals, y_vals, marker=marker_list[ii], c=color_list[ii], alpha=0.7, label=lin_label)
        plt.plot(x_vals, ypred, ls=line_list[ii], c=color_list[ii], alpha=0.7, label=r2_label)
    
    # Reset the order of labels in the plot legend to a common sequence between graphs.
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend([handles[idx] for idx in leg_order],[labels[idx] for idx in leg_order])
    
    # Stores the graphs as .png files for use outside of the script, then displays the graph for the user.
    pic_title = g_title + '.png'
    pic_title = pic_title.replace(' ', '_')
    if not os.path.isdir('./graphs'):
        os.mkdir('graphs')
    file_path = './graphs/' + pic_title
    plt.savefig(file_path)
    plt.show()
    
    
    
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
    
    
    
def gen_plots(init_df, zero_df, strain_id, edited_data=False):

    """
    gen_plots(init_df, zero_df, strain_id, edited_data=False)
    
    Using processed data frames (init_df, zero_df) from a specific strain (strain_id), generates a standardized set of graphs for
    visual data analysis.  The graph titles can also be modified to represent if the data frame values have been edited
    (edited_data=True).
    """
    
    if edited_data:
        end_phrase = ' Survival - Edited'
    else:
        end_phrase = ' Survival'

    
    graph_title = 'Effects of Freezing on ' + strain_id + ' IR' + end_phrase
    graph_list = [['Aq', 'RT'], ['Aq', -80]]

    data_display(zero_df, graph_title, graph_list)

    graph_title = 'Effects of Desiccation on ' + strain_id + ' IR' + end_phrase
    graph_list = [['De', 'RT'], ['De', -80]]

    data_display(zero_df, graph_title, graph_list)

    graph_title = 'Environmental Effects on ' + strain_id + ' IR' + end_phrase
    graph_list = [['Aq', 'RT'], ['Aq', -80], ['De', 'RT'], ['De', -80]]

    data_display(zero_df, graph_title, graph_list)

    graph_title = 'IR and Environmental Effects on ' + strain_id + end_phrase
    graph_list = [['Aq', 'RT'], ['Aq', -80],['De', 'RT'], ['De', -80]]

    data_display(init_df, graph_title, graph_list, use_init=True)
    
    graph_title = 'Effects of Desiccation Duration on ' + strain_id + ' IR' + end_phrase
    graph_list = [['De', 'RT'], ['De', -80],['D28-De', 'RT'], ['D28-De', -80]]

    data_display(zero_df, graph_title, graph_list)
    
    graph_title = 'Effects of Desiccation Duration on ' + strain_id + end_phrase
    graph_list = [['De', 'RT'], ['De', -80],['D28-De', 'RT'], ['D28-De', -80]]

    data_display(init_df, graph_title, graph_list, use_init=True)
    
    
    
def analyze_cfu_counts(file_loc, plate_dilute, count_sel=2, remove_values=False, removal_list=None):
    
    """
    analyze_cfu_counts(file_name, plate_dilute, count_sel=2, remove_values=False, removal_list=None)
    
    Main function to call to perform data analysis on formatted excel files containing CFU counts.  Function will
    default to second plate read unless otherwise specified.  Files with only one plate count should not trigger
    first-vs-second plate count decision.
    
    Arguments:
    
    - file_loc (type str):  Location and name of excel file to be processed.  User must specify either full or
      relative path to excel file to be processed.
    
    - plate_dilute (type float):  Dilution factor of the plating method used.  Necessary to get absolute CFU counts
      for each data cell.  Best way to determine is (uLs put on plate / 1000 uL)
    
    - count_sel (type int, default 2):  For files with two plate counts, number of the plate count to be analyzed.
      Will default to the second count unless otherwise specified.  count_sel is not used for files with only one
      count.
    
    - remove_values (type bool, default False):  Informs function if certain data values need to be removed from the
      final processed data.  If True, requires a list passed to removal_list to function.
      
    - removal_list (type list of tuples, default None):  List of tuples defining the location of data values that
      need to be removed from the final processed data.  remove_values must be True for this list to function.
    
    Returns:
    
    Function returns three objects: init_processed_df, zero_processed_df, strain_name
    
    - init_processed_df:  data frame of log10 survival data normalized to the initial concentration

    - zero_processed_df:  data frame of log10 survival data normalized to the control (0 kGy reading) for each condition
    
    - strain_name:  string containing the species and strain of the microorganism used to generate the data; useful for
      future labeling functions
    
    Function also returns individual .csv files of processed data for each ['Condition', 'Temp'] set, saved in the
    './archive'.  If
    remove_values is True, resulting files will be marked as "Edited".
    
    Returned dataframe format example:
    
        Temp                -80                  RT          
                    mean       std      mean       std
    Condition                                            
    Aq        0.0 -0.806700  0.486300 -0.063881  0.194777
              1.0       NaN       NaN -0.056648  0.601776
              2.0 -0.659301  0.419340 -0.644220  0.268310
              3.0       NaN       NaN -0.866339  0.279646
              4.0 -1.140718  0.601953 -1.044173  0.175404
    """

    # These variables will be used to identify excel files with two plate readings and determine which set to use.
    two_readings = [20, 24]
    dup_cuts = [7, 20]
    trip_cuts = [9, 24]
    sec_set_toggle = count_sel

    # Read the first sheet of the excel file as a dataframe.
    master_df = pd.read_excel(file_loc, sheet_name=0, header=None, na_values=['Contaminant', 'Over'])

    # Basic error check to ensure submitted excel file has the right formatting.
    if master_df.iloc[1:2, 0:5].count().sum() > 0:
        raise ValueError('Input Excel file format incorrect. Please ensure input file cells are formatted correctly.')

    # Extract and format the species and strain data from the dataframe for later use.
    species = str(master_df.iloc[3, 1])
    strain = str(master_df.iloc[3, 2])
    strain_name = species + ' ' + strain
    file_strain_name = strain_name.replace('. ', '-').replace(' ', '_')

    # If the excel file has two readings, sets data truncation points according to whether the data has duplicates or
    # triplicates.
    if master_df.index.size in two_readings:
        if master_df.index.size in dup_cuts:
            trim_index = dup_cuts
            triplicate = False
        elif master_df.index.size in trip_cuts:
            trim_index = trip_cuts
            triplicate = True
    
    # Truncate the dataframe according to which reading will be used.  See 'local_functions.py' for details on
    # trim_excel().
        master_df = trim_excel(master_df, trim_index, sec_set_toggle)
    
    # Add colum and index labels according to cells in the dataframe, and remove extraneous cells.  See
    # 'local_functions.py' for details on clean_excel().
    
    cleaned_df = clean_excel(master_df)
    cleaned_df = cleaned_df.replace(0, np.nan)

    # Generate conversion factors to convert data values to concentrations (CFU/mL), incorporating plate_dilute.
    dilutions = cleaned_df.columns.get_level_values(2)
    plate_factor = np.log10(plate_dilute)
    conversions = 1 / (10 ** (dilutions + plate_factor))
    converted_df = cleaned_df.mul(conversions, axis='columns')

    # Calculate average initial concentrations for each triplicate/duplicate.
    converted_df['Init_Con_Avg'] = converted_df['Init_Con'].mean(axis=1)

    # Normalize all values to 'Init_Con_Avg', then bring 'Dilutions' into the Index for future .groupby().
    converted_df = converted_df.apply(lambda x: x / (converted_df['Init_Con_Avg']))
    init_norm_df = converted_df.drop(columns=['Init_Con', 'Init_Con_Avg'], level=0)
    init_norm_df = init_norm_df.stack(0)

    # Branch point for calculations done normalized to initial concentration (init_norm_df) and normalized to control
    # (zero dose, zero_norm_df).
    zero_norm_df = init_norm_df

    # Normalizes zero_norm_df to control (0 kGy dose), yielding new_zero_norm_df.  See 'local_functions.py' for details
    # on zero_normalize().
    new_zero_norm_df = zero_normalize(zero_norm_df)

    # Process data for graphing and analysis.  See 'local_functions.py' for details on data.calc().
    init_processed_df = data_calc(init_norm_df)
    zero_processed_df = data_calc(new_zero_norm_df)

    # If Day 6 desiccated values ('D6-De') are present in the data, performs a conversion on these values and adds them
    # to the Day 5 ('De') data set.
    if 'D6-De' in zero_processed_df.index.get_level_values(0):
        day_6_conv(zero_processed_df)
    
    # If remove_values is set to True, remove values from the final data sets using values from removal_list.  See
    # 'local_functions.py' for details on value_removal().
    if remove_values:
        init_processed_df = value_removal(init_processed_df, removal_list)
        zero_processed_df = value_removal(zero_processed_df, removal_list)
        file_strain_name = "Edited_" + file_strain_name
    
    # Exports output dataframes into .csv files by ('Condition', 'Temp') label for archival purposes.  See
    # 'local_functions.py' for details on csv_export().
    csv_export(init_processed_df, 'init', file_strain_name)
    csv_export(zero_processed_df, 'zero', file_strain_name)

    return init_processed_df, zero_processed_df, strain_name