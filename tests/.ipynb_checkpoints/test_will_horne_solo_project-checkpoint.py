from will_horne_solo_project import __version__
from will_horne_solo_project import local_functions as lf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
import pytest

def test_version():
    assert __version__ == '0.1.0'

@pytest.fixture
def loaded_file(): 
    '''Load file for testing.'''
    file_loc = 'S-cerevisiae_EXF-6761.xlsx'
    master_df = pd.read_excel(file_loc, sheet_name=0, header=None, na_values=['Contaminant', 'Over'])
    return master_df

@pytest.fixture
def trimmed_file(loaded_file):
    '''Gets loaded file to trimmed stage of main function; test effect of trim_excel().'''
    trim_index = [9, 24]
    sec_set_toggle = 2
    triplicate = True
    trimmed_df = lf.trim_excel(loaded_file, trim_index, sec_set_toggle)
    return trimmed_df

@pytest.fixture
def cleaned_file(trimmed_file):
    '''Gets loaded file to cleaned_df stage of main function; test effect of clean_excel().'''
    cleaned_df = lf.clean_excel(trimmed_file)
    cleaned_df = cleaned_df.replace(0, np.nan)
    return cleaned_df

@pytest.fixture
def converted_file(cleaned_file):
    '''Gets loaded file to converted_df stage of main function; test plate count to concentration conversion.'''
    dilutions = cleaned_file.columns.get_level_values(2)
    plate_factor = np.log10(0.1)
    conversions = 1 / (10 ** (dilutions + plate_factor))
    converted_df = cleaned_file.mul(conversions, axis='columns')
    return converted_df
    
def test_read_blank_space(loaded_file):
    """
    If file loads correctly from read_excel(), there should be a space of Nan values in the data frame from (1,0) to (2, 5).
    The .count() method should, therefore, return 0 for this region.
    """
    assert loaded_file.iloc[1:2, 0:5].count().sum() == 0
    assert 0

def test_name_extraction(loaded_file):
    """
    Tests code responsible for generating strings for file names and graph titles to ensure it extracts and formats the required
    strings from the data frame properly.
    """
    species = loaded_file.iloc[3, 1]
    strain = loaded_file.iloc[3, 2]
    strain_name = species + ' ' + strain
    file_strain_name = strain_name.replace('. ', '-').replace(' ', '_')
    assert strain_name == 'S. cerevisiae EXF-6761'
    assert file_strain_name == 'S-cerevisiae_EXF-6761'
    assert 0
    
def test_trimming(trimmed_file):
    """
    Tests that trim_excel() creates a data frame of the right size.
    """
    assert len(trimmed_file.index) == 9
    assert 0

def test_cleaning(cleaned_file, trimmed_file):
    """
    Tests that clean_excel() creates a data frame of the expected height and width.
    """
    assert len(trimmed_file.index) == (len(cleaned_file.index) + 3)
    assert len(trimmed_file.columns) == (len(cleaned_file.columns) + 6)
    assert 0
    
def test_conversion(cleaned_file, converted_file):
    """
    Tests that specific colony count values from the cleaned_file data frame were converted to the correct concentration in the
    converted_file data frame.
    """
    test_locs = [[0,0,10000000], [4,7,100000], [3,49,1000000]]
    for loc_v in test_locs:
        assert converted_file.iloc[loc_v[0],loc_v[1]] == (cleaned_file.iloc[loc_v[0],loc_v[1]] * loc_v[2])
        assert converted_file.iloc
        assert converted_file.iloc