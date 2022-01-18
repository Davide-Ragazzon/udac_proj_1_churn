# library doc string


import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import logging

logging.basicConfig(
    filename='./results.log',
    level=logging.INFO,
    filemode='a',
    #format='%(name)s - %(levelname)s - %(message)s',
    format='%(asctime)s %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

logger = logging.getLogger(os.path.basename(__file__))


def import_data(pth):
    '''
    Returns dataframe for the csv found at pth.
    It makes sure that there is a Churn column with values 0 or 1.

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    df = pd.read_csv(pth)
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    return df


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''

    fig_churn = plt.figure(figsize=(20, 10))
    ax = fig_churn.gca()
    df['Churn'].hist(ax=ax)

    fig_age = plt.figure(figsize=(20, 10))
    ax = fig_age.gca()
    df['Customer_Age'].hist(ax=ax)

    fig_marital_status = plt.figure(figsize=(20, 10))
    ax = fig_marital_status.gca()
    df.Marital_Status.value_counts('normalize').plot(kind='bar', ax=ax)

    fig_tot_trans_ct = plt.figure(figsize=(20, 10))
    ax = fig_tot_trans_ct.gca()
    sns.distplot(df['Total_Trans_Ct'], ax=ax)

    fig_corr = plt.figure(figsize=(20, 10))
    ax = fig_corr.gca()
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2, ax=ax)
    pass


def encoder_helper(df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    proportion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    pass


def perform_feature_engineering(df, response):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    pass


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    pass


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    pass
