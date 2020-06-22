# Disaster Response Pipeline Project
This project trains a machine learning pipeline for categorisation of messages during a disaster. The model categorises an input message from an emergency worker so that we can forward the information to respective units. A web app is deployed using Flask, which loads the pre trained model and categorises a user-input message.

## Table of Contents
1. [Dependencies](#deps)
2. [Project Motivation](#motivation)
3. [Dataset] (#dataset)
4. [File Descriptions](#desc)
5. [Results](#results)
6. [Instructions](#instructions)

## Dependencies<a name="deps"></a>
Pandas 1.0.1, numpy 1.17.4, scikit-learn 0.20.0
The code is developed using python3 (3.7.0), and the above libraries. It should probably run on a different version of the above libraries too.

## Project Motivation<a name="motivation"></a>
During a natural calamity/ disaster various government agencies and NGOs come together to help the victims and mitigate the situation. During these times, processing of emergency messages from people/ social workers into respective categories could be helpful, for re-directing the message to the concerned authorities/ organisations. Hence, this project aims to build a model to categorise input emergency messages into pre-defined categories.

## Dataset<a name="dataset"></a>
Credits to Figure Eight (https://appen.com/) and Udacity for putting together this dataset. On a whole, it contains around 25K messages categorised into a total of 36 categories.

## File Descriptions<a name="desc"></a>
data/*
The folder contains two csv files corresponding to messages and their corresponding categories. In file `process_data.py`, we clean the data and merge these two dataset into one, containing the input message and one-hot encoding of the categories it belongs to. We then store the dataframe in an sql database, `DisasterResponse.db`.

models/*
In file `train_classifier.py`, we train a multi-class classification pipeline using Random Forest Classifier and Grid Search to find the best parameters. The input text is tokenized using commonly used techniques such as CountVectorizer and tf-idf transformation. Latent Semantic Analysis is performed on this for dimensionality reduction.
We train a pipeline, display classification metrics on the test data and save the best fit model to `classifier.pkl`

## Results<a name="results"></a>
The average f-score of the classifier is around 0.5, with a precision of 0.6. The model is able to classify the given input message as relevant or not with a f-score of 0.86. For dominant categories, the f-score is around 0.6, lie Aid, weather related, food.


## Instructions<a name="instructions"></a>
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
