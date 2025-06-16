# imports and settings

from argparse import ArgumentParser
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error
import joblib
import os
from datetime import datetime
import time
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
# from rulefit import RuleFit
from os.path import join
from os import system

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

# auxiliary functions
# Definições dos modelos e flags
# MODEL_FILE = "old_models/random_forest_3.5F_model_10.79.pkl"
MODEL_FILE = "filtered-dataset-models/random_forest_model_12.60_mean_withrtt_mergeclsv.pkl"

# MODEL_FILE_STD = "old_models/random_forest_model_stdevs_max_f_7.89.pkl"
MODEL_FILE_STD = "filtered-dataset-models/random_forest_model_stdevs_9.31_with_rtt_mergeclsv.pkl"

# MODEL_FILE_WO_TR = "old_models/random_forest_model_10.31_wo_alldata.pkl"
# MODEL_FILE_WO_TR = "random_forest_model_7.18_mean_wortt2.pkl"
MODEL_FILE_WO_TR = "filtered-dataset-models/random_forest_model_12.60_mean_withrtt_mergeclsv.pkl"

# MODEL_FILE_WO_TR_STD = "old_models/random_forest_model_stdevs_7.95_wo_alldata.pkl"
# MODEL_FILE_WO_TR_STD = "random_forest_model_stdevs_6.41_wortt.pkl"
MODEL_FILE_WO_TR_STD = "filtered-dataset-models/random_forest_model_stdevs_9.31_with_rtt_mergeclsv.pkl"


def model_extract_rulefit(model_path: str):
    """
    Given a string that is the path to a model,
    returns its rules extracted
    """

    # load model
    model_data = joblib.load(model_path)
    model = model_data['model']

    # apply extraction
    model_extract = RuleFit(model)
    rules = model_extract.get_rules()
    print(rules)
    print(type(rules))
    exit()

    # TODO: verify what is the format of the rules
    return rules

def models_extract_rulefit(model_paths: str):
    """
    Given a list of paths to models,
    returns its rules extracted
    """

    # df to hold rules for each model
    models_rules_df_list = []

    # looping over models
    for model_path in model_paths:
        model_rules = model_extract_rulefit(model_path)

    pass


def save_rules():
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
    model_extract_rulefit(model_path=input_path)


######################################################################
# running main function


if __name__ == '__main__':
    main()

######################################################################
# end of current module





