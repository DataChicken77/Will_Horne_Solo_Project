import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
import os

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