import sys

import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Loads two pandas dataframes from the given input files and returns a merged dataframe.

    INPUTS:
    messages_filepath - csv file containing messages data
    categories_filepath - csv file containing category data

    RETURNS:
    A pandas dataframe containing messages and its corresponding category in one hot
    encoded format.
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on=['id'])
    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[:1]
    category_colnames = list(row.apply(lambda x: (x.str.split('-'))[0][0]))
    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories[column].apply(lambda x: x.split('-')[1])
        categories[column] = categories[column].astype('int')
        categories[column] = categories[column].apply(lambda x: 1 if x > 1 else x)
    df = df.drop(columns=['categories'])
    df = pd.merge(df, categories, left_index=True, right_index=True)
    return df

def clean_data(df):
    '''
    Removes duplicates from the dataframe.
    '''
    df['dup'] = df.duplicated(subset=['id'], keep='first')
    df = df[~df['dup']]
    df = df.drop(columns=['dup'])
    return df


def save_data(df, database_filename):
    '''
    Creates an sqlite database containing the dataframe, in the given file.
    '''
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('tweets', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
