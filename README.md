# Disaster Response Pipeline

## Table of Contents
1. [Context](#context)
2. [Objectives](#objectives)
3. [Structure](#structure)
4. [Installation](#installation)
5. [Dependencies](#dependencies)
6. [Authors](#authors)
7. [License](#license)
8. [Acknowledgements](#acknowledgement)
9. [Screenshots](#screenshots)

<a name="context"></a>
## Context

This project analyzes disaster data to build a model for an API that classifies disaster messages. The project contains real messages that were sent during disaster events. The purpose of this program is to categorize these events using machine learning pipeline to send the messages to an appropriate disaster relief agency.

<a name="objectives"></a>
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

<a name="structure"></a>
## Structure

Here's the file structure of the project

```
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
```

<a name="installation"></a>
## Installation

- Clone the project
  ```
  https://github.com/MKasigwa/disaster_response_pipeline.git
  ```
- Change to the new disaster_response_pipeline directory : cd disaster_response_pipeline
- To run ETL pipeline and clean data :
  * Change to data directory  :
    ```
    cd data
    python ./process_data.py
    ```
- To run ML pipeline that loads data and classifier
    ```
    cd models
    python ./train_classifier.py
    ```
-  Start the app
  * Change to the app directory : 
    ```
    cd app
    python run.py
    ```
- Go to http://127.0.0.1:3001

<a name="dependencies"></a>
## Dependencies
- Python 3.5+
- Machine Learning Libraries: NumPy, SciPy, Pandas, Sciki-Learn
- Natural Language Process Libraries: NLTK
- SQLlite Database Libraqries: SQLalchemy
- Model Loading and Saving Library: Pickle
- Web App and Data Visualization: Flask, Plotly

<a name="authors"></a>
## Authors

- [Miracle Kasigwa](https://github.com/MKasigwa)

<a name="license"></a>

## License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<a name="acknowledgement"></a>
## Acknowledgements

[Udacity](https://www.udacity.com/) Data Science Nanodegree Program

<a name="screenshots"></a>
## Screenshots
