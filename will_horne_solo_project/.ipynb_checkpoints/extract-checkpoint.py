import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
import os

import clean as cln
import display as disp
import process as proc

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

    # Import necessary local modules.
    import clean as cln
    import display as disp
    import process as proc
    
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
        master_df = cln.trim_excel(master_df, trim_index, sec_set_toggle)
    
    # Add colum and index labels according to cells in the dataframe, and remove extraneous cells.  See
    # 'local_functions.py' for details on clean_excel().
    
    cleaned_df = cln.clean_excel(master_df)
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
    new_zero_norm_df = proc.zero_normalize(zero_norm_df)

    # Process data for graphing and analysis.  See 'local_functions.py' for details on data.calc().
    init_processed_df = proc.data_calc(init_norm_df)
    zero_processed_df = proc.data_calc(new_zero_norm_df)

    # If Day 6 desiccated values ('D6-De') are present in the data, performs a conversion on these values and adds them
    # to the Day 5 ('De') data set.
    if 'D6-De' in zero_processed_df.index.get_level_values(0):
        proc.day_6_conv(zero_processed_df)
    
    # If remove_values is set to True, remove values from the final data sets using values from removal_list.  See
    # 'local_functions.py' for details on value_removal().
    if remove_values:
        init_processed_df = cln.value_removal(init_processed_df, removal_list)
        zero_processed_df = cln.value_removal(zero_processed_df, removal_list)
        file_strain_name = "Edited_" + file_strain_name
    
    # Exports output dataframes into .csv files by ('Condition', 'Temp') label for archival purposes.  See
    # 'local_functions.py' for details on csv_export().
    csv_export(init_processed_df, 'init', file_strain_name)
    csv_export(zero_processed_df, 'zero', file_strain_name)

    return init_processed_df, zero_processed_df, strain_name