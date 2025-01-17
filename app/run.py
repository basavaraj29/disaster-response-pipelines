import json
import plotly
import pandas as pd

import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.stem.porter import PorterStemmer

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine

import re
nltk.download(['punkt', 'wordnet'])

app = Flask(__name__)

url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
def tokenize(text):
    '''
    Function to tokenize input text.
    '''
    text = re.sub(url_regex, ' ', text)
    text = word_tokenize(re.sub(r'[^a-zA-Z0-9]', ' ', text))
    stem = PorterStemmer()
    return [stem.stem(word.lower().strip()) for word in text]

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('tweets', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    cols = set(df.columns).difference(set(['id', 'message', 'original', 'genre']))
    X = []
    Y = []
    for col in cols:
        X.append(col)
        Y.append((len(df[df[col] == 1])*100.0)/len(df))

    tmp_df = df[['genre', 'id']].groupby('genre').count()
    tmp_df['id'] = tmp_df['id']/len(df)

    graphs = [
        {
            'data': [
                Bar(
                    x=X,
                    y=Y
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Percent"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=list(tmp_df.index),
                    y=list(tmp_df['id'])
                )
            ],

            'layout': {
                'title': 'Distribution of Message Sources',
                'yaxis': {
                    'title': "Percent"
                },
                'xaxis': {
                    'title': "Source"
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
