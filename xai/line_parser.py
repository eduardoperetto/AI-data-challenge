# imports and settings

import os
import re
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
def parse_line(line):
    """
    this function takes a line (str) based on the rulefit algorithm output
    and parse it to obtain a dictionary of the information wanted
    """
    match = re.match(r"^(.*) \(type=(linear|rule), coef=([-\d.eE]+), imp=([-\d.eE]+)\)$", line.strip())
    if match:
        feature = match.group(1)
        type_ = match.group(2)
        coef = float(match.group(3))
        imp = float(match.group(4))
        return {"rule": feature, "type": type_, "coefficient": coef, "importance": imp}
    else:
        print(f"Warning: line did not match expected format:\n{line}")
        return None

def convert_txt_to_csv(input_txt_path,
                       output_csv_path
                       ) -> None:
    """
    this function takes a .txt file path and
    parses it to convert it into a .csv
    """
    # reading txt file
    with open(input_txt_path, 'r') as file:
        lines = file.readlines()

    # parsing file lines
    parsed = [parse_line(line) for line in lines]
    parsed = [p for p in parsed if p is not None]

    # turning output dict into df
    df = pd.DataFrame(parsed)

    # sorting dataframe by rule importance
    df.sort_values(by="importance", ascending=False, inplace=True)

    # saving output df
    df.to_csv(output_csv_path, index=False)
    print(f"CSV written to: {output_csv_path}")


def make_models_rule_df(rules_folder_path:str,
                        rule_file_extension:str,
                        output_folder:str
                        ) -> None:
    """
    given a directory containing models, a directory containing
    the x_data and a directory containing the predictions
    returns a csv of extracted rules and fidelity metrics
    of all models
    """

    # getting model files in respective input folder
    rule_files = get_files_in_folder(path_to_folder=rules_folder_path,
                                     extension=rule_file_extension)

    # iterating through df
    for rule_file in rule_files:

        # getting current model input path
        rules_input_path = join(rules_folder_path,
                                rule_file)

        # create a model id to trace from which csv it came from
        model_id = rule_file.split('_')[0]

        # create the path to save the files
        rules_output_path = join(output_folder,
                                 f'{model_id}_parsed_output.csv')

        # run parser in file
        convert_txt_to_csv(rules_input_path, rules_output_path)

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
    description = 'organizes a .txt file outputed by the rulefit algorithm and stores it into a csv'

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser

    # input models directory path
    parser.add_argument('-i', '--input_folder',
                        dest='input_folder',
                        required=True,
                        help='defines path to input directory containing .txt rules')

    # input models directory path
    parser.add_argument('-x', '--files-extension',
                        dest='files_extension',
                        required=True,
                        help='defines rules files extension type')

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
    input_folder = args_dict['input_folder']

    # getting model input path
    files_extension = args_dict['files_extension']

    # getting images extension
    output_folder = args_dict['output_folder']

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # waiting for user input
    enter_to_continue()

    # running function to preprocess images in a folder
    make_models_rule_df(rules_folder_path=input_folder,
                        rule_file_extension=files_extension,
                        output_folder=output_folder)

######################################################################
# running main function


if __name__ == '__main__':
    main()

######################################################################
# end of current module





