# Train the model and save it in a pickle file
# python train_classifier.py --database_filename ../../db.sqlite3 --model_pickle_filename trained_classifier.pkl --grid_search_cv

import os
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import GridSearchCV
import pickle
import argparse


MODEL_PICKLE_FILENAME = 'trained_classifier.pkl'
DATABASE_FILENAME = '../db.sqlite3'
TABLE_NAME = 'disaster_message'


def get_df_from_database(database_filename):
    '''
    Return dataframe from the database

    Args:
        database_filename (str): database filename. Default value DATABASE_FILENAME

    Returns:
        df (pandas.DataFrame): dataframe containing the data 
    '''
    engine = create_engine('sqlite:///' + database_filename)
    return pd.read_sql_table(TABLE_NAME, engine)


def load_data(database_filename):
    '''
    Load the data from the database

    Args:
        database_filename (str): database filename. Default value DATABASE_FILENAME

    Returns:
        X (pandas.Series): dataset
        Y (pandas.DataFrame): dataframe containing the categories
        category_names (list): list containing the categories name
    '''
    df = get_df_from_database(database_filename) 
    X = df['message']
    Y = df.iloc[:, 4:]
    category_names = list(df.columns[4:])
    return X, Y, category_names


def tokenize(text):
    '''
    Clean and tokenize the text in input

    Args:
        text (str): input text

    Returns:
        clean_tokens (list): tokens obtained from the input text
    '''
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, 'urlplaceholder')

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model(grid_search_cv = False):
    '''
    Build the model

    Args:
        grid_search_cv (bool): if True after building the pipeline it will be performed an exhaustive search over specified parameter values ti find the best ones

    Returns:
        pipeline (pipeline.Pipeline): model
    '''
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer = tokenize)), 
                     ('tfidf', TfidfTransformer()), 
                     ('clf', MultiOutputClassifier(RandomForestClassifier()))
                    ])

    #pipeline.get_params()

    if grid_search_cv == True:
        print('Searching for best parameters...')
        parameters = {'vect__ngram_range': ((1, 1), (1, 2))
            , 'vect__max_df': (0.5, 0.75, 1.0)
            , 'tfidf__use_idf': (True, False)
            , 'clf__estimator__n_estimators': [50, 100, 200]
            , 'clf__estimator__min_samples_split': [2, 3, 4]
        }

        pipeline = GridSearchCV(pipeline, param_grid = parameters)

    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluate the model performances and print the results

    Args:
        model (pipeline.Pipeline): model to evaluate
        X_test (pandas.Series): dataset
        Y_test (pandas.DataFrame): dataframe containing the categories
        category_names (str): categories name
    '''
    Y_pred = model.predict(X_test)
    # Calculate the accuracy for each of them.
    for i in range(len(category_names)):
       print('Category: {} '.format(category_names[i]))
       print(classification_report(Y_test.iloc[:, i].values, Y_pred[:, i]))
       print('Accuracy {}\n\n'.format(accuracy_score(Y_test.iloc[:, i].values, Y_pred[:, i])))


def save_model(model, model_filename):
    '''
    Save in a pickle file the model

    Args:
        model (pipeline.Pipeline): model to be saved
        model_pickle_filename (str): destination pickle filename
    '''
    pickle.dump(model, open(model_filename, 'wb'))


def load_model(model_pickle_filename):
    '''
    Return model from pickle file

    Args:
        model_pickle_filename (str): source pickle filename

    Returns:
        model (pipeline.Pipeline): model readed from pickle file 
    '''
    return pickle.load(open(model_pickle_filename, 'rb'))


def parse_input_arguments():
    '''
    Parse the command line arguments

    Returns:
        database_filename (str): database filename. Default value DATABASE_FILENAME
        model_pickle_filename (str): pickle filename. Default value MODEL_PICKLE_FILENAME
        grid_search_cv (bool): If True perform grid search of the parameters
    '''
    parser = argparse.ArgumentParser(description = "Disaster Response Pipeline Train Classifier")
    parser.add_argument('--database_filename', type = str, default = DATABASE_FILENAME, help = 'Database filename of the cleaned data')
    parser.add_argument('--model_pickle_filename', type = str, default = MODEL_PICKLE_FILENAME, help = 'Pickle filename to save the model')
    parser.add_argument('--grid_search_cv', action = "store_true", default = False, help = 'Perform grid search of the parameters')
    args = parser.parse_args()
    #print(args)
    return args.database_filename, args.model_pickle_filename, args.grid_search_cv


def train(database_filename, model_pickle_filename, grid_search_cv = False):
    '''
    Train the model and save it in a pickle file

    Args:
        database_filename (str): database filename
        model_pickle_filename (str): pickle filename
        grid_search_cv (bool): if True after building the pipeline it will be performed an exhaustive search over specified parameter values ti find the best ones

    '''
    # print(database_filename)
    # print(model_pickle_filename)
    # print(grid_search_cv)
    # print(os.getcwd())

    print('Download nltk componets if needed...')
    nltk.download(['punkt', 'wordnet'])

    print('Loading data...\n    Database: {}'.format(database_filename))
    X, Y, category_names = load_data(database_filename)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

    print('Building model...')
    model = build_model(grid_search_cv)

    print('Training model...')
    model.fit(X_train, Y_train)

    print('Evaluating model...')
    evaluate_model(model, X_test, Y_test, category_names)

    print('Saving model...\n    Model: {}'.format(model_pickle_filename))
    save_model(model, model_pickle_filename)

    print('Trained model saved!')


if __name__ == '__main__':
    database_filename, model_pickle_filename, grid_search_cv = parse_input_arguments()
    train(database_filename, model_pickle_filename, grid_search_cv)