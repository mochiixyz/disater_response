import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle

nltk.download(['wordnet', 'punkt', 'stopwords'])

def load_data(database_filepath):
    """
       Load data from database into frame

       Input:
       database_filepath: the path of the database

       Output:
       X: Message
       Y: Category of disaster
       """
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('disaster_messages_tbl', engine)
    X = df['message']
    Y = df.iloc[:, 4:]
    return X, Y


def tokenize(text):
    """
    Tokenize text

    Input:
      text: message
    Output:
      lemm: list of tokens
    """

    #Normalize
    text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)

    #Convert text into tokens
    words = word_tokenize(text)

    #Remove stopwords
    words = [word for word in words if word not in stopwords.words("english")]

    #Lemmatization
    lemm = [WordNetLemmatizer().lemmatize(w) for w in words]

    return lemm


def build_model():
    """
     Model to classify the disaster messages

     Ouput:
       cv: classification model
     """

    #Build pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    #Parameter selection using GridSearchCV
    parameters = {
        'tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [50, 60, 70]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test):
    """
    Evaluate model
    
    Inout:
    model: classification model
    X_test: test dataset of X
    Y_test: test dataset of Y
    """

    Y_pred = model.predict(X_test)
    report = classification_report(Y_test, Y_pred, target_names=Y_test.columns)
    print(report)


def save_model(model, model_filepath):
    """
    Saves model in pickle file

    Input:
    model: classification model
    model_filepath: path where model saved
    """

    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

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