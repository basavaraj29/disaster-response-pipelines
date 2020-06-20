import sys

import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.stem.porter import PorterStemmer

import pandas as pd

from sqlalchemy import create_engine

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn.externals import joblib

import re
nltk.download(['punkt'])

url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

def load_data(database_filepath):
    '''
    Loads pandas dataframe from the given sqlite database file and retuns list of
    inout messages, output categories and categories list.
    '''

    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table(table_name='tweets', con=engine)
    X = df['message']
    Y = df.drop(columns=['id', 'message', 'original', 'genre'], axis=1)
    category_names = list(Y.columns)
    return X, Y, category_names

def tokenize(text):
    '''
    Function to tokenize input text.
    '''
    text = re.sub(url_regex, ' ', text)
    text = word_tokenize(re.sub(r'[^a-zA-Z0-9]', ' ', text))
    stem = PorterStemmer()
    return [stem.stem(word.lower().strip()) for word in text]


def build_model():
    '''
    Returns a pipeline for multi-class classification for text input
    '''
    pipeline = Pipeline([ \
        ('vect', CountVectorizer(tokenizer=tokenize)), \
        ('tfidf', TfidfTransformer()), \
        ('svd', TruncatedSVD(n_components=20, random_state=42)), \
        ('clf', MultiOutputClassifier(RandomForestClassifier(random_state=42))) \
    ])
    return pipeline

def evaluate_model(model, X_test, y_test, category_names):
    '''
    Prints classification report for test data on each category
    '''
    y_pred = model.predict(X_test)
    for i in range(len(category_names)):
        col = y_test.columns[i]
        print(col)
        print(classification_report(y_test[col].tolist(), y_pred[:,i:i+1]))
    return


def save_model(model, model_filepath):
    '''
    Writes the model to the given filepath
    '''
    joblib.dump(model, model_filepath, compress=1)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
