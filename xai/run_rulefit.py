# imports and settings

import os
import csv
import joblib
import te2rules
import numpy as np
import pandas as pd
from os import system
from os.path import join
from sklearn.tree import _tree
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

# TODO: get y_pred and X
def model_extract_rules(model_path: str,
                        data_x: pd.DataFrame,
                        data_y_pred: pd.DataFrame) -> tuple:
    """
    Given a string that is the path to a model,
    returns its rules extracted
    """

    # load model
    model_data = joblib.load(model_path)
    model = model_data['model']

    # get feature names
    feature_names = model.feature_names_in_

    # apply extraction
    model_explainer = ModelExplainer(model=model,
                                     feature_names=feature_names,
                                     verbose=True
                                    )

    rules_list = model_explainer.explain(X=data_x,
                                         y=data_y_pred,
                                         num_stages=10,  # stages can be between 1 and max_depth
                                         min_precision=0.95,  # higher min_precision can result in rules with more terms overfit on training data
                                         jaccard_threshold=0.5  # lower jaccard_threshold speeds up the rule exploration, but can miss some good rules
                                        )


    fidelity, positive_fidelity, negative_fidelity = model_explainer.get_fidelity()

    return rules_list, fidelity, positive_fidelity, negative_fidelity


def models_extract_rules(models_folder_path:str,
                         extension:str,
                         output_folder:str
                         ) -> None:
    """
    given a directory containing models
    returns a csv of extracted rules and fidelity metrics
    """

    # getting masks files in respective input folder
    model_files = get_files_in_folder(path_to_folder=models_folder_path,
                                      extension=extension)

    # create empty list to hold the rule dfs
    rules_dfs_list = []

    # create empty list to hold the rule dfs
    fidelity_dfs_list = []

    # iterating over mask files
    for file_index, model_file in enumerate(model_files, 1):

        # getting current model input path
        model_input_path = join(models_folder_path,
                                model_file)

        # get model rules df
        rules_list,  fidelity, positive_fidelity, negative_fidelity = model_extract_rules(model_path=)

        # converting list of rules into df
        rules_df = pd.DataFrame({'rules': rules_list})

        # registering model on rules df
        rules_df['model'] = model_file

        # put fidelity metrics into dict
        extraction_metrics_dict = {'fidelity': fidelity,
                                   'positive_fidelity': positive_fidelity,
                                   'negative_fidelity': negative_fidelity,
                                   'model': model_file
                                  }

        extraction_metrics_df = pd.DataFrame(extraction_metrics_dict, index=[0])

        # append current rule df to dfs list
        rules_dfs_list.append(rules_df)

        # append current fidelity df to dfs list
        fidelity_dfs_list.append(extraction_metrics_dict)

    # concatenating "dfs" from dfs lists into
    # a pandas dataframe
    # of the rules
    models_rule_df = pd.concat(rules_dfs_list, ignore_index=True)

    # of the fidelities
    models_rule_df = pd.concat(rules_dfs_list, ignore_index=True)

    # create the path to save the files
    rules_output_path = join(output_folder,
                       'models_rules.csv')
    fidelity_output_path = join(output_folder,
                       'models_fidelities.csv')

    # saving dfs
    models_rule_df.to_csv(rules_output_path)
    models_rule_df.to_csv(fidelity_output_path)

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
    PartialDependenceDisplay.from_estimator(clf, X, features)
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

    # input csv path
    parser.add_argument('-i', '--input-path',
                        dest='input_path',
                        required=True,
                        help='defines path to input directory containing models')

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

    # getting input csv path
    input_path = args_dict['input_path']

    # getting images extension
    output_folder = args_dict['output_folder']

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # waiting for user input
    enter_to_continue()

    # running function to preprocess images in a folder
    model_extract_rules(model_path=input_path)


######################################################################
# running main function


if __name__ == '__main__':
    main()

######################################################################
# end of current module





