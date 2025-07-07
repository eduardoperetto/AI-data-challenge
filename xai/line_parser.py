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
def clean_line(line: str) -> str:
    """
    Given a parsed txt line,
    returns clean line.
    """
    # getting clean line
    line = " ".join(line.split())

    # returning clean line
    return line


def txt_to_csv(input_path: str,
               output_path: str
               ) -> None:
    """
    Given a weird ass text file, parses
    lines, and saves file as .csv.
    """
    # defining placeholder for parsed lines
    parsed_lines = []

    # reading file
    with open(input_path, 'r') as open_file:

        # reading lines
        lines = open_file.readlines()

        # iterating over lines
        for line_index, line in enumerate(lines):

            # cleaning line
            line = clean_line(line=line)

            # checking line index
            if line_index == 0:  # means it's header (no need to remove index)

                # splitting line
                line_split = line.split()

                # assembling parsed line (joining by comma)
                parsed_line = ','.join(line_split)

                # adding enter to end of line
                parsed_line += '\n'

                # appending current line to parsed lines
                parsed_lines.append(parsed_line)

            else:  # means it's not header (needs to remove index)

                # splitting line
                line_split = line.split()

                # discarding index
                line_split = line_split[1:]

                # getting last four elements (no spaces within elements)
                line_end = line_split[-4:]

                # getting first elements (might contain spaces within element)
                line_start = line_split[:-4]

                # joining elements from line start (joining by space)
                line_start = ' '.join(line_start)

                # assembling line list
                line_list = [line_start]
                line_list.extend(line_end)

                # assembling parsed line (joining by comma)
                parsed_line = ','.join(line_list)

                # adding enter to end of line
                parsed_line += '\n'

                # appending current line to parsed lines
                parsed_lines.append(parsed_line)

    # writing file
    with open(output_path, 'w') as open_file:

        # writing lines
        open_file.writelines(parsed_lines)


def make_models_rule_df(input_folder_path:str,
                        file_extension:str,
                        output_folder_path:str
                        ) -> None:
    """
    given a directory containing models, a directory containing
    the x_data and a directory containing the predictions
    returns a csv of extracted rules and fidelity metrics
    of all models
    """

    # getting model files in respective input folder
    rule_files = get_files_in_folder(path_to_folder=input_folder_path,
                                     extension=file_extension)

    # iterating through df
    for rule_file in rule_files:

        # getting current model input path
        rules_input_path = join(input_folder_path,
                                rule_file)

        # create a model id to trace from which csv it came from
        model_id = rule_file.split('_')[0]

        # create the path to save the files
        output_path = join(output_folder_path,
                           f'{model_id}_parsed_output.csv')

        # run parser in file
        txt_to_csv(rules_input_path, output_path)

    return

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
    make_models_rule_df(input_folder_path=input_folder,
                        file_extension=files_extension,
                        output_folder_path=output_folder)

######################################################################
# running main function


if __name__ == '__main__':
    main()

######################################################################
# end of current module





