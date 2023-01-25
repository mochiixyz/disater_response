import json
import plotly
import pandas as pd
import warnings
import pickle
import os

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
# from sklearn.externals import joblib
import sklearn.externals
import joblib

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('disaster_messages_tbl', engine)

# load model
model_path = os.path.realpath('../models/classifier.pkl')
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=UserWarning)
    model = joblib.load(model_path,'rb')

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    genre_related = df[df['related']==1].groupby('genre').count()['message']
    genre_not_related = df[df['related']==0].groupby('genre').count()['message']
    genre_name_plot2 = list(genre_related.index)

    genre_request = df[df['request']==1].groupby('genre').count()['message']
    genre_not_request = df[df['request']==0].groupby('genre').count()['message']
    genre_name_plot3 = list(genre_request.index)

    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data' : [
                Bar(
                    x = genre_name_plot2,
                    y = genre_related,
                    name = 'Related'
                ),
                Bar(
                    x = genre_name_plot2,
                    y = genre_not_related,
                    name = 'Not related'
                )
            ],
            'layout': {
                'title': 'Distribution of message by genre and related',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data' : [
                Bar(
                    x = genre_name_plot3,
                    y = genre_request,
                    name = 'Request'
                ),
                Bar(
                    x = genre_name_plot3,
                    y = genre_not_request,
                    name = 'Not request'
                )
            ],
            'layout': {
                'title': 'Distribution of message by genre and request',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON = graphJSON)

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
    app.run(host='0.0.0.0', port=3000, debug=True)

if __name__ == '__main__':
    main()
