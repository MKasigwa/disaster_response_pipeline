# Disaster Response Pipeline

## Context

This project analyzes disaster data to build a model for an API that classifies disaster messages. The project contains real messages that were sent during disaster events. The purpose of this program is to categorize these events using machine learning pipeline to send the messages to an appropriate disaster relief agency.

## Objectives

This project have three important processes :

1. ETL Pipeline Process : the data cleaning pipeline

- Loads the messages and categories datasets;
- Merges the two datasets
- Cleans the data
- Stores clean data in a SQLite database

2. ML Pipeline Process : the machine learning pipeline

- Loads data from the SQLite database;
- Splits the dataset into training and test sets;
- Builds a text processing and machine learning pipeline;
- Trains and tunes a model using GridSearchCV
- Outputs results on the rest set
- Exports the final model as a pickle file

3. Flask Web App : The data visualizations frontEnd.

## Structure

Here's the file structure of the project

- app
  | - templates
  | | - base.html # the base html template of the web app
  | | - master.html # the main page of web app. It extends the base template
  | | - go.html # classification result page of web app. It extends the base template
  | - run.py # Flask file that runs app

- data
  | - disaster_categories.csv # data to process
  | - disaster_messages.csv # data to process
  | - process_data.py # data cleaning pipeline
  | - DisasterMessages.db # database to save clean data to

- models
  | - train_classifier.py # machine learning pipeline
  | - classifier.pkl # saved model

- README.md

## Installation

- Clone the project https://github.com/MKasigwa/disaster_response_pipeline.git
- Change to the new disaster_response_pipeline directory : cd disaster_response_pipeline
- Change to the app directory : cd app
- Start the app : python run.py
  - Running on http://127.0.0.1:3001

<a name="authors"></a>

## Authors

- [Miracle Kasigwa](https://github.com/MKasigwa)

<a name="license"></a>

## License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
