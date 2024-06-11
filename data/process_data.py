# Process the data and save it in a database
# python process_data.py --messages_filename disaster_messages.csv --categories_filename disaster_categories.csv --database_filename ../../db.sqlite3

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

CATEGORIES_FILENAME = 'disaster_categories.csv'
MESSAGES_FILENAME = 'disaster_messages.csv'
DATABASE_FILENAME = './InsertDatabaseName.db'
TABLE_NAME = 'disaster_message'


def load_message_categories(disaster_messages_file, disaster_categories_file):
    '''
    Load the messages and categories from csv files and merge it to the dataframe

    Args:
        disaster_messages_file (str): disaster categories filepath
        disaster_categories_file (str): disaster categories filepath

    Returns:
        new merged dataframe:  containing messages and categories
    '''
    messages = pd.read_csv(disaster_messages_file)
    categories = pd.read_csv(disaster_categories_file)
    df = pd.merge(messages, categories, on = 'id')
    return df


def clean_data(df):
    '''
    This function cleans data from dataframe

    Args:
        df : dataframe with the not clean data

    Returns:
        clean dataframe : (dataframe)
    '''
    #load categories data
    categories = pd.read_csv('./disaster_categories.csv')
    # create a dataframe of the 36 individual category columns
    categories = pd.Series(categories['categories']).str.split(pat=';', n=-1, expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    # use this row to extract a list of new column names for categories.
    category_colnames = row.apply(lambda x: x.split('-')[0])    
    # rename the columns of `categories`
    categories.columns = category_colnames
    # set each value to be the last character of the string
    for column in categories:
        categories[column] = categories[column].apply(lambda x:int(x.split('-')[1]))
    # convert column from string to numeric
    categories[column] = categories[column].apply(pd.to_numeric)
    # drop the original categories column from `df`
    df.drop('categories',axis='columns', inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df.reset_index(drop=True), categories.reset_index(drop=True)], axis=1)
    # check number of duplicates
    num_duplicates = df.duplicated().sum()
    print('Number of duplicated is {}'.format(num_duplicates))
    # drop duplicates
    df = df.drop_duplicates()
    # check number of duplicates
    num_duplicates = df.duplicated().sum()
    print('Number of duplicated is {}'.format(num_duplicates))
    return df
    

def save_data_to_database(df):
    '''
    Save the data into the database.

    Args:
        df : dataframe
    '''
    engine = create_engine('sqlite:///InsertDatabaseName.db')
    df.to_sql('disaster_message', engine, index = False)  


def parse_input_arguments():
    '''
    Parse the command line arguments

    Returns:
        categories_filename (str): categories filename. Default value CATEGORIES_FILENAME
        messages_filename (str): messages filename. Default value MESSAGES_FILENAME
        database_filename (str): database filename. Default value DATABASE_FILENAME
    '''
    parser = argparse.ArgumentParser(description = "Disaster Response Pipeline Process Data")
    parser.add_argument('--messages_filename', type = str, default = MESSAGES_FILENAME, help = 'Messages dataset filename')
    parser.add_argument('--categories_filename', type = str, default = CATEGORIES_FILENAME, help = 'Categories dataset filename')
    parser.add_argument('--database_filename', type = str, default = DATABASE_FILENAME, help = 'Database filename to save cleaned data')
    args = parser.parse_args()
    #print(args)
    return args.messages_filename, args.categories_filename, args.database_filename


def process(messages_filename, categories_filename, database_filename):
    '''
    Process the data and save it in a database

    Args:
        categories_filename (str): categories filename
        messages_filename (str): messages filename
        database_filename (str): database filename
    '''
    # print(messages_filename)
    # print(categories_filename)
    # print(database_filename)
    # print(os.getcwd())

    print('Loading data...\n    Messages: {}\n    Categories: {}'.format(messages_filename, categories_filename))
    df = load_data(messages_filename, categories_filename)

    print('Cleaning data...')
    df = clean_data(df)

    print('Saving data...\n    Database: {}'.format(database_filename))
    save_data(df, database_filename)

    print('Cleaned data saved to database!')


if __name__ == '__main__':

    #Loading merged dataframe 
    df = load_message_categories('./disaster_messages.csv', './disaster_categories.csv');
    #clean data
    df = clean_data(df)

    # Save the clean dataset into an sqlite database.
    save_data_to_database(df)