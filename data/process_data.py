# Process the data and save it in a database
# python process_data.py --messages_filename disaster_messages.csv --categories_filename disaster_categories.csv --database_filename ../../db.sqlite3

import sys
import pandas as pd
from sqlalchemy import create_engine

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
    df = pd.merge(messages,categories, on='id')
    return df


def clean_data(df, categorie_file_path):
    '''
    This function cleans data from dataframe

    Args:
        df : dataframe with the not clean data
        categorie_file_path : categorie file path

    Returns:
        clean dataframe : (dataframe)
    '''
    #load categories data
    categories = pd.read_csv(categorie_file_path)
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
        categories[column] = categories[column].astype(str).str[-1:]
        # convert column from string to numeric
        categories[column] = categories[column].apply(pd.to_numeric)
    # drop the original categories column from `df`
    df.drop('categories',axis='columns', inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories],join='inner', axis=1)
    # check number of duplicates
    num_duplicates = df.duplicated().sum()
    print('Number of duplicated is {}'.format(num_duplicates))
    # drop duplicates
    df = df.drop_duplicates()
    # check number of duplicates
    num_duplicates = df.duplicated().sum()
    print('Number of duplicated is {}'.format(num_duplicates))
    return df
    

def save_data_to_database(df, db_path):
    '''
    This function Saves clean data into the database.
    Args:
        df : dataframe
    '''
    engine = create_engine('sqlite:///' + db_path)
    df.to_sql('disaster_message', engine, index=False, if_exists='replace')

if __name__ == '__main__':
    if len(sys.argv) >= 4:
        #Loading merged dataframe 
        print('#Loading merged dataframe')
        message_file_path, categories_file_path , database_file_path =sys.argv[1:] #'./disaster_messages.csv'
        #categories_file_path = sys.argv[2] #'./disaster_categories.csv'
        #database_file_path = sys.argv[3] #DisasterMessages.db
        df = load_message_categories(message_file_path, categories_file_path)
        #clean data
        print('#Cleaning data')
        df = clean_data(df,categories_file_path)
        # Save the clean dataset into an sqlite database.
        print('#Saving clean dataset into an sqlite database')
        save_data_to_database(df,database_file_path)
    else:
        print('Please provide message, categories and database file paths')