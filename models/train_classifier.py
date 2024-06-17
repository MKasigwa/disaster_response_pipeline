# Train the model and save it in a pickle file
# python train_classifier.py --database_filename ../../db.sqlite3 --model_pickle_filename trained_classifier.pkl --grid_search_cv
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.multioutput import MultiOutputClassifier
import pickle
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import classification_report, accuracy_score


def get_df_from_database(database_file_path):
    '''
    Return dataframe from the database

    Args:
        database_file_path (str): database file path.

    Returns:
        df : dataframe containing the data 
    '''
    engine = create_engine('sqlite:///' + database_file_path)
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
class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    Starting Verb Extractor class
    
    This class extract the starting verb of a sentence,
    creating a new feature for the ML classifier
    """

    # Starting verb method
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
    new_pipeline = Pipeline([
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

def save_model(pipeline, pickle_filepath):
    """
    This function saves Pipeline
    
    Arguments:
        pipeline -> pipeline to be saved
        pickle_filepath -> destination path to save .pkl file
    """
    pickle.dump(pipeline, open(pickle_filepath, 'wb'))


if __name__ == '__main__':
    if len(sys.argv) >= 3:
    # load data from database
        print('Loading data from database')
        database_file_path, model_file_path =sys.argv[1:] #'../data/DisasterMessages.db'
        #model_file_path =sys.argv[2] #'classifier.pkl'
        df = get_df_from_database(database_file_path)
        X = df['message']
        Y = df.iloc[:,4:]
        category_names = list(df.columns[4:])

        #Build a machine learning pipeline
        print('Building the pipeline')
        pipeline = build_pepeline()
        
        #Train pipeline
        print('Training the pipeline ')
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
        pipeline = pipeline.fit(X_train, Y_train)

        print('Classifing the pipeline ')
        y_pred = pipeline.predict(X_test)

        # Print the classification report
        print("Print the classification report")    
        for i in range(len(category_names)):
            print('Category {}: {} '.format(i, category_names[i]))
            print(classification_report(Y_test.iloc[:, i].values, y_pred[:, i]))
            print('Accuracy {}\n\n'.format(accuracy_score(Y_test.iloc[:, i].values, y_pred[:, i])))

        #Instantiate GridSearchCV
        print('Instantiating GridSearchCV')
        parameters = {
            'features__text_pipeline__count_vectorizer__ngram_range': ((1, 1), (1, 2)),  # Example of tuning ngram_range
            'classifier__estimator__n_estimators': [50, 100, 200],  # Example of tuning AdaBoostClassifier's n_estimators
            'classifier__estimator__learning_rate': [0.1, 0.5, 1.0]  # Example of tuning AdaBoostClassifier's learning_rate
        }
        cv = GridSearchCV(pipeline, param_grid=parameters)

        #Train pipeline
        print('Training the GridSearchCv ')
        cv.fit(X_train, Y_train)

         # Calculate classification report
        print('Calculate classification report for GridSearchCv')
        y_pred = cv.predict(X_test)

        # Print classification report on test data
        print("Printing the classification report")   
        print(classification_report(Y_test, y_pred))
        
        #Saved model
        print("Saving the model")   
        save_model(pipeline,model_file_path)
    else:
        print('Please provide database and model file paths')