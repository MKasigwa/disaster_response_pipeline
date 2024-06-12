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
from sklearn.multioutput import MultiOutputClassifier
import pickle
from scipy.stats import gmean
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
import argparse


MODEL_PICKLE_FILENAME = 'trained_classifier.pkl'
DATABASE_FILENAME = '../db.sqlite3'
TABLE_NAME = 'disaster_message'


def get_df_from_database(database_filepath):
    '''
    Return dataframe from the database

    Args:
        database_filename (str): database filename. Default value DATABASE_FILENAME

    Returns:
        df : dataframe containing the data 
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    return pd.read_sql_table('disaster_message', engine)

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
       text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# Build a custom transformer which will extract the starting verb of a sentence
class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    Starting Verb Extractor class
    
    This class extract the starting verb of a sentence,
    creating a new feature for the ML classifier
    """

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    # Given it is a tranformer we can return the self 
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

def build_pepeline():
    '''
    Build pipeline

    Output: 
        pipeline : pipeline
    '''
    new_pipeline = pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('count_vectorizer', CountVectorizer(tokenizer=tokenize, token_pattern=None)),
                ('tfidf_transformer', TfidfTransformer())
            ])),

            ('starting_verb_transformer', StartingVerbExtractor())
        ])),

        ('classifier', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    return new_pipeline

def multioutput_fscore(y_true,y_pred,beta=1):
    """
    MultiOutput Fscore
    
    This is a performance metric of my own creation.
    It is a sort of geometric mean of the fbeta_score, computed on each label.
    
    It is compatible with multi-label and multi-class problems.
    It features some peculiarities (geometric mean, 100% removal...) to exclude
    trivial solutions and deliberatly under-estimate a stangd fbeta_score average.
    The aim is avoiding issues when dealing with multi-class/multi-label imbalanced cases.
    
    It can be used as scorer for GridSearchCV:
        scorer = make_scorer(multioutput_fscore,beta=1)
        
    Arguments:
        y_true -> List of labels
        y_prod -> List of predictions
        beta -> Beta value to be used to calculate fscore metric
    
    Output:
        f1score -> Calculation geometric mean of fscore
    """
    
    # If provided y predictions is a dataframe then extract the values from that
    if isinstance(y_pred, pd.DataFrame) == True:
        y_pred = y_pred.values
    
    # If provided y actuals is a dataframe then extract the values from that
    if isinstance(y_true, pd.DataFrame) == True:
        y_true = y_true.values
    
    f1score_list = []
    for column in range(0,y_true.shape[1]):
        score = fbeta_score(y_true,y_pred,beta,average='weighted')
        f1score_list.append(score)
        
    f1score = np.asarray(f1score_list)
    f1score = f1score[f1score<1]
    
    # Get the geometric mean of f1score
    f1score = gmean(f1score)
    return f1score

def save_model(pipeline, pickle_filepath):
    """
    This function saves Pipeline
    
    Arguments:
        pipeline -> GridSearchCV or Scikit Pipelin object
        pickle_filepath -> destination path to save .pkl file
    """
    pickle.dump(pipeline, open(pickle_filepath, 'wb'))


if __name__ == '__main__':
    # load data from database
    print('Loading data from database')
    df = get_df_from_database('../data/InsertDatabaseName.db')
    X = df['message']
    Y = df.iloc[:,4:]
    category_names = list(df.columns[4:])

    #Build a machine learning pipeline
    print('Building the pipeline')
    pipeline = build_pepeline()
    
    #Train pipeline
    print('Training the pipeline ')
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
    pipeline.fit(X_train, Y_train)

    save_model(pipeline,'classifier.pkl')