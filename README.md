# Processing Plate Count Data from Microorganisms Irradiated under Different Environmental Extremes

My research focuses on how environmental effects (such as drying/desiccating and deeply freezing) can potentially improve the ionizing radiation survival of various microorganisms.  As part of this study, I irradiated, at different radiation doses (measured in kilo-Greys, kGy), triplicate or duplicate preparations of several microorganisms (one *E. coli* strain, three *Bacillus* strains, two *S. cerevisiae* strains, one *A. baumannii* strain) in aqueous (liquid media), aqueous frozen (-80oC), desiccated, and desiccated frozen conditions.  I then plated the irradiated samples and counted either colony-forming-units (CFUs) or spores (for *Bacillus* strains).  I stored these counts in spreadsheet files that were formatted for human readability (merged cells for column banners, specific text for contaminated or overgrown plates, etc.).  Also, for some of the strains, I took multiple plate reads and put the second counts on the same sheet as the first counts.

The goal of this project is to perform automated data analysis of these files with the following outputs:
* Log average values of CFU or spore population decrease, normalized to both the initial culture concentration and the control (0 kGy) dose for each condition
* Plotted regression curves of each condition set, merged as needed to promote comparisons
* Statistical analysis of the regressions to determine significance of any differences in the regression slopes (signal for improved survival)

I plan to accomplish these goals through the following steps:
* Read the spreadsheet (Excel file) into a pandas DataFrame
* Format and tidy the DataFrame for data analysis, using multiindex rows and columns
* Conduct data processing (initial and zero normalizations, averaging, logarithmic conversion)
* Perform data analysis (linear and/or quadratic regression, statistical tests of coefficients)
* Graph and display data in a logical, human-readable format
* Present findings in an orderly manner (Jupyter Notebook)

## Installation instructions

+ Install the conda packaging manager

+ Run the following commands in bash:

```
conda create -n solo_pro python=3 pip
git clone https://github.com/DataChicken77/Will_Horne_Solo_Project.git
cd Will_Horne_Solo_Project
pip install -e .
```


## Usage


```
import will_horne_solo_project as whs
## Add some examples here of using your project:

input_file = 'path_to_excel.xls'
whs.clean_excel(input_file)
...
```


## Reading Excel into Pandas DataFrame:

All Excel files containing my research data are formatted in the pattern displayed below:

![alt text](https://github.com/DataChicken77/class_project/blob/master/pictures/data_file_example.JPG "Data File Example")

To start the build of my main function, `analyze_cfu_counts()`, I used the `pd.read_excel()` function to import the data into `master_df`.  However, I had to build a data file storage directory (IR_Datasets) to help organize the raw data.  Also, I added additional constraints to `read_excel()` to set ‘Over’ and ‘Contaminant’ as Nan values; it’s good to have a record of these values in the raw data, but they aren’t required in the actual data analysis.

As an aside, I originally built this script with a `request_file()` function that would generate a GUI for the user to select a file for analysis.  When I turned my main code into a function (`analyze_cfu_count()`), I no longer needed this prompt; however, `request_file()` is still available if needed in a future version.

## Format and Tidy DataFrame:

Because I used the above Excel format throughout my research to promote uniformity of data display and assist with data entry, I had some established ground rules for how `master_df` would look when it was loaded.  The first cleaning step I used was identifying when a file had two plate reads instead of one.  This was done primarily through row count; files with two counts had a larger, predetermined number of rows compared to files with only one.  By default, I have the main function use the second plate read; however, the function can use the first plate read if `count_sel=1` is passed.  With the proper read selected, `trim_excel()` shortened `master_df` to only the rows needed.

Once the proper plate reading was selected, `clean_excel()` went to work.  It started by removing the column headers and index listings (identified by relying on the pre-formatted nature of the data file), reformatting them so that pandas could recognize them as index and column labels, then reattaching them to the data.  I implemented some basic error checking at this step to verify if the index and column data frames are formatted correctly, and prompt the user to review and fix the data file if they are not.  Finally, I converted the data values to numeric (some were still strings from the initial reading) for later processing.

## Conduct Data Processing:

Once the data was formatted properly, I began processing it by converting the plate CFU counts to CFU concentrations (CFUs/mL), using values from the ‘Dilution’ level of the column headers.  Because I used different plating methods for different data sets, I factored in the dilution of the plating method using the passed value `plate_dilute`.  I then used the ‘Init_Conc’ columns to create average initial concentrations for each triplicate/duplicate, then normalized all other values to these averages.

At this point, I created a duplicate data frame, giving me one that would only be normalized to the initial concentration (`init_norm_df`) and one that would be normalized to both initial concentration and the zero dose for each condition set (`zero_norm_df`).  I then sequentially normalized each condition set in `zero_norm_df` using `zero_normalize()`.

With the two data frames normalized, I used `data_calc()` to perform the final calculations, converting all values to log10, then averaging across dilutions and triplicates/duplicates to get the mean value and standard deviation for each dose in each condition set.  For record-keeping purposes, I ported the values for each condition set into .csv files named for each strain, condition set, and normalization method (i.e. A-baumannii_AB5075_D28-De_-80_zero.csv) using `csv_export()`.

During testing, I realized that I had a data file that would require additional processing.  During my *A. baumannii* experiment, I underestimated its radiation survival while in the desiccated state.  I had to irradiate additional samples the next day at higher doses to assess the limits of its survival.  I added a 'Day 6' value to the library I used to convert the top-level column values ('Condition'), then built the `day_6_conv()` function to process the additional data.  When used on the data frame returned from `data_calc()`, it finds the highest common dose between 'De' and 'D6_De' data values, uses it to create a conversion factor for the additional 'D6_De' values, then converts these values and appends them to the 'De' data values.

As part of our lab's initial analysis of the data, we noted that plate counts taken at the very limit of survival, where colonies were present only in undiluted ('Dilution' == 0) samples, created a tail effect on the data.  We wanted to see how the data would look with these "tailing" values removed; therefore, I built the `value_removal()` function.  This function removes a list of values, passed as a list of tuples identifying the location of the values, from the data frames resulting from `data_calc()` and `day_6_conv()`.  I then set the main function with additional passed values (`remove_values` and `removal_list`) to selectively use `value_removal()`.

## Perform Data Analysis and Display Results:

I realized that, for my results and analysis, I would need to have modularity at the condition set level so I could compare different condition sets directly.  I developed `data_display()` to help specify what condition sets went onto each individual graph.  By passing in a list of condition sets (‘Condition’ and ‘Temp’ values in list format) with a processed data frame and a graph title, `data_display()` generated a scatter plot of data points, error bars for each data point, and a regression curve (linear or quadratic) to fit the data, for each condition set passed.  To generate the regression curves, I used functions from the `statsmodels` and `sklearn` python packages.

I also knew, however, that I would have some standardized presentation methods I would use across each microorganism.  I set up `gen_plots()` to create this standardize set of figures, plotting aqueous vs. aqueous frozen, desiccated vs. desiccated frozen, and all four conditions together.  I also graphed a zoomed-in version of all four conditions from the init processed data sets to demonstrate any change in CFU concentrations caused solely by the acts of freezing or desiccation.

## References/Resources Used:

"pandas 0.25.3 documentation." *pandas: powerful Python data analysis toolkit.* https://pandas.pydata.org/pandas-docs/stable/index.html. Copyright (c) 2008-2012, AQR Capital Management, LLC, Lambda Foundry, Inc. and PyData Development Team. All rights reserved.

Hunter, JD. "Matplotlib: A 2D Graphics Environment." *Computing in Science & Engineering,* vol. 9, no. 3, pp. 90-95, 2007.  Reference at https://matplotlib.org/index.html.

"NumPy."  https://numpy.org/index.html.  Copyright © 2005-2019, NumPy Developers.  All rights reserved.

Seabold, Skipper, and Josef Perktold. “Statsmodels: Econometric and statistical modeling with python.” *Proceedings of the 9th Python in Science Conference.* 2010.  Reference at https://www.statsmodels.org/stable/index.html.

Pedregosa et al. "Scikit-learn: Machine Learning in Python." *JMLR* 12, pp. 2825-2830, 2011.  Reference at https://scikit-learn.org/stable/index.html.

Too many "Stack Overflow" articles to count.  https://stackoverflow.com/.