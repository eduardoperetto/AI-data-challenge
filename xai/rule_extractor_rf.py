# imports and settings

import os
import csv
import joblib
import te2rules
import itertools
import numpy as np
import pandas as pd
from os import system
from os.path import join
from sklearn.tree import _tree, export_text
from xgboost import XGBRegressor
from argparse import ArgumentParser
from te2rules.explainer import ModelExplainer
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay

##########################################################################################
#aux-aux functions :p
def spacer(char: str = '_',
           reps: int = 50
           ) -> None:
    """
    Given a char and a number of reps,
    prints a "spacer" string assembled
    by multiplying char by reps.
    :param char: String. Represents a character to be used
    as basis for spacer.
    :param reps: Integer. Represents number of character's
    repetitions used in spacer.
    :return: None.
    """
    # defining spacer string
    spacer_str = char * reps

    # printing spacer string
    print(spacer_str)


def print_execution_parameters(params_dict: dict) -> None:
    """
    Given a list of execution parameters,
    prints given parameters on console,
    such as:
    '''
    --Execution parameters--
    input_folder: /home/angelo/Desktop/ml_temp/imgs/
    output_folder: /home/angelo/Desktop/ml_temp/overlays/
    '''
    :param params_dict: Dictionary. Represents execution parameters names and values.
    :return: None.
    """
    # defining base params_string
    params_string = f'--Execution parameters--'

    # iterating over parameters in params_list
    for dict_element in params_dict.items():
        # getting params key/value
        param_key, param_value = dict_element

        # adding 'Enter' to params_string
        params_string += '\n'

        # getting current param string
        current_param_string = f'{param_key}: {param_value}'

        # appending current param string to params_string
        params_string += current_param_string

    # printing final params_string
    spacer()
    print(params_string)
    spacer()

def enter_to_continue():
    """
    Waits for user input ("Enter")
    and once press, continues to run code.
    """
    # defining enter_string
    enter_string = f'press "Enter" to continue'

    # waiting for user input
    input(enter_string)

def get_files_in_folder(path_to_folder: str,
                        extension: str
                        ) -> list:
    """
    Given a path to a folder, returns a list containing
    all files in folder that match given extension.
    """
    # getting all files in folder
    all_files_in_folder = os.listdir(path_to_folder)

    # getting specific files
    files_in_dir = [file  # getting file
                    for file  # iterating over files
                    in all_files_in_folder  # in input folder
                    if file.endswith(extension)]  # only if file matches given extension

    # sorting list
    files_in_dir = sorted(files_in_dir)

    # returning list
    return files_in_dir

########################################################################### auxiliary functions
def make_running_df(models_folder_path:str,
                    x_data_folder_path:str,
                    models_extension:str,
                    x_data_extension:str
                    ) -> pd.DataFrame:
    # getting model files in respective input folder
    model_files = get_files_in_folder(path_to_folder=models_folder_path,
                                      extension=models_extension)

    # getting data files in respective input folder
    x_data_files = get_files_in_folder(path_to_folder=x_data_folder_path,
                                       extension=x_data_extension)

    # create list to append and organize files
    model_dfs_list = []
    x_data_dfs_list = []

    for model_file, x_data_file in zip(model_files, x_data_files):
        # make the file names into df so we can organize the runs
        model_df = pd.DataFrame({'id': [model_file.split('_')[0]],
                                 'model_file': [model_file]})
        x_data_df = pd.DataFrame({'id': [x_data_file.split('_')[0]],
                                  'x_data_file': [x_data_file]})

        # append current rule df to dfs list
        model_dfs_list.append(model_df)
        x_data_dfs_list.append(x_data_df)

    # concatenating to have the full df
    models_df = pd.concat(model_dfs_list, ignore_index=True)
    x_data_dfs = pd.concat(x_data_dfs_list, ignore_index=True)

    # merging to make the triplets of model, x_data and predictions
    paired_df = pd.merge(models_df, x_data_dfs, how="inner", on="id")

    return paired_df


# Defining function to extract rules from a decision tree
def get_rules_tree(tree, feature_names):
    tree_ = tree.tree_  # Getting the underlying tree structure
    # Creating a list of feature names, replacing undefined features with a placeholder string
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    paths = []  # List to store all the paths (rules) in the tree
    path = []  # Temporary list to store the current path

    # Recursive function to traverse the tree and collect rules
    def recurse(node, path, paths):
        # If the node is not a leaf node
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            p1, p2 = list(path), list(path)
            p1 += [f"({name} <= {np.round(threshold, 6)})"]
            recurse(tree_.children_left[node], p1, paths)
            p2 += [f"({name} > {np.round(threshold, 6)})"]
            recurse(tree_.children_right[node], p2, paths)
        else:
            # Regression output is a single scalar value
            value = float(tree_.value[node][0][0])
            samples = int(tree_.n_node_samples[node])
            path += [(value, samples)]
            paths += [path]

    # Starting the recursive function from the root node
    recurse(0, path, paths)

    # Sorting the paths based on the number of samples
    samples_count = [p[-1][1] for p in paths]
    ii = list(np.argsort(samples_count))
    paths = [paths[i] for i in reversed(ii)]

    rules = []  # List to store the rules in a formatted way
    for path in paths:
        rule = "if "

        for p in path[:-1]:
            if rule != "if ":
                rule += " and "
            rule += str(p)

        predicted_value = path[-1][0]
        samples = path[-1][1]

        # Appending the formatted rule to the rules list
        rules.append({
            'rule': rule,
            'prediction': predicted_value,
            'samples': samples
        })

    return rules


def get_rules_forest(model_path:str,
                     x_data_path:str
                     )-> pd.DataFrame:
    # load model
    model_data = joblib.load(model_path)
    model = model_data['model']

    # load x data
    x_data = pd.read_csv(x_data_path)

    # Getting feature names from the DataFrame df_train, excluding the target variable column
    feature_names = x_data.columns

    # Creating an empty list to store the rules from each tree
    rules_list = []

    # Iterating through each decision tree in the random forest to collect rules
    for tree in model.estimators_:
        rules = get_rules_tree(tree, feature_names)
        rules_list.extend(rules)

    # Converting the list of dictionaries into a DataFrame
    rules_df = pd.DataFrame(rules_list)

    rules_df.reset_index(drop=True, inplace=True)

    return rules_df



def models_extract_rules(models_folder_path:str,
                         x_data_folder_path:str,
                         models_extension:str,
                         x_data_extension:str,
                         output_folder:str
                         ) -> None:
    """
    given a directory containing models, a directory containing
    the x_data and a directory containing the predictions
    returns a csv of extracted rules and fidelity metrics
    of all models
    """

    paired_df = make_running_df(models_folder_path=models_folder_path,
                                 x_data_folder_path=x_data_folder_path,
                                 models_extension=models_extension,
                                 x_data_extension=x_data_extension
                                 )

    # create empty list to hold the rule dfs
    rules_dfs_list = []

    # iterating through df
    for index, row in paired_df.iterrows():

        # getting current model input path
        model_input_path = join(models_folder_path,
                                row['model_file'])

        # getting current x_data input path
        x_data_input_path = join(x_data_folder_path,
                                row['x_data_file'])

        # get model rules df
        rules_df = get_rules_forest(model_path=model_input_path,
                                    x_data_path=x_data_input_path
                                    )

        # registering model on rules df
        rules_df['model'] = row['model_file']

        # append current rule df to dfs list
        rules_dfs_list.append(rules_df)

    # concatenating "dfs" from dfs lists into
    # a pandas dataframe
    # of the rules
    models_rule_df = pd.concat(rules_dfs_list, ignore_index=True)

    # create the path to save the files
    rules_output_path = join(output_folder,
                       'models_rules.csv')

    # saving dfs
    models_rule_df.to_csv(rules_output_path)

    # printing execution message
    print(f'output saved to {output_folder}')
    print('analysis complete!')

    return


def get_pdp_model(model_path: str,
                  data_x: pd.DataFrame):
    """
    plot and save partial dependence plots for features
    """
    # load model
    model_data = joblib.load(model_path)
    model = model_data['model']
    # TODO: partial dependence plots
    # PartialDependenceDisplay.from_estimator(clf, X, features)
    pass
#####################################################################
# argument parsing related functions


def get_args_dict() -> dict:
    """
    Parses the arguments and returns a dictionary of the arguments.
    """
    # defining program description
    description = 'runs a Rule Extraction Algorithm (RuleFit) that works in Ensembles'

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser

    # input models directory path
    parser.add_argument('-m', '--models-input-path',
                        dest='models_input_path',
                        required=True,
                        help='defines path to input directory containing models')

    # input x_data directory path
    parser.add_argument('-x', '--x-datas-input-path',
                        dest='x_datas_input_path',
                        required=True,
                        help='defines path to input directory containing the x_data')

    # input models directory path
    parser.add_argument('-mx', '--model-extension',
                        dest='model_extension',
                        required=True,
                        help='defines model files extension type')

    # input models directory path
    parser.add_argument('-xx', '--x-data-extension',
                        dest='x_data_extension',
                        required=True,
                        help='defines x_data files extension type')

    # output folder
    parser.add_argument('-o', '--output-folder',
                        dest='output_folder',
                        required=True,
                        help='defines path to output files')

    # creating arguments dictionary
    args_dict = vars(parser.parse_args())

    # returning the arguments dictionary
    return args_dict


######################################################################
# defining main function


def main():
    """Runs main code."""
    # getting args dict
    args_dict = get_args_dict()

    # getting model input path
    models_input_path = args_dict['models_input_path']

    # getting model input path
    x_datas_input_path = args_dict['x_datas_input_path']

    # getting model input path
    model_extension = args_dict['model_extension']

    # getting model input path
    x_data_extension = args_dict['x_data_extension']

    # getting images extension
    output_folder = args_dict['output_folder']

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # waiting for user input
    enter_to_continue()

    # running function to preprocess images in a folder
    models_extract_rules(models_folder_path=models_input_path,
                         x_data_folder_path=x_datas_input_path,
                         models_extension=model_extension,
                         x_data_extension=x_data_extension,
                         output_folder=output_folder
                         )

######################################################################
# running main function


if __name__ == '__main__':
    main()

######################################################################
# end of current module





