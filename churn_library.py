""" Library that predicts churners

"""

import logging
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split

import constants as const

logging.basicConfig(
    filename=const.RESULTS_LOG,
    level=logging.INFO,
    filemode='w',
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
    logger.info("Loading data")
    df = pd.read_csv(pth)
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    return df


def save_into_folder(fig, fig_name, folder=const.IMG_FOLDER):
    """Utility function for saving figures as .png files and logging what happened.

    Args:
        fig: figure to save
        fig_name: name used when saving
        folder (optional): folder where the figure needs to be saved.
            Defaults to the image folder as defined in the constants.py as IMG_FOLDER.
    """
    logger.info(f"Saving {fig_name}.png")
    file_png = os.path.join(folder, f"{fig_name}.png")
    fig.savefig(file_png)
    # Added because having the figure always displayed can get very annoying...
    plt.close(fig)


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    logger.info("**** Performing EDA")

    logger.info("Plotting distribution of Churn")
    fig = plt.figure(figsize=(20, 10))
    ax = fig.gca()
    df['Churn'].hist(ax=ax)
    save_into_folder(fig, 'eda_churn')

    logger.info("Plotting distribution of Customer_Age")
    fig = plt.figure(figsize=(20, 10))
    ax = fig.gca()
    df['Customer_Age'].hist(ax=ax)
    save_into_folder(fig, 'eda_age')

    logger.info("Plotting distribution of Marital_Status")
    fig = plt.figure(figsize=(20, 10))
    ax = fig.gca()
    df.Marital_Status.value_counts('normalize').plot(kind='bar', ax=ax)
    save_into_folder(fig, 'eda_marital_status')

    logger.info("Plotting distribution of Total_Trans_Ct")
    fig = plt.figure(figsize=(20, 10))
    ax = fig.gca()
    # Causes a future warning but decided it is ok for this exercise
    sns.distplot(df['Total_Trans_Ct'], ax=ax)
    save_into_folder(fig, 'eda_tot_trans_ct')

    logger.info("Plotting correlation between features")
    fig = plt.figure(figsize=(20, 10))
    ax = fig.gca()
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2, ax=ax)
    save_into_folder(fig, 'eda_corr')


def encoder_helper(df, category_lst, encoded_category_lst=None):
    """Helper function to turn each categorical column into a new column with
    proportion of churn for each category - associated with cell 15 from the notebook

    Args:
        df: pandas dataframe
        category_lst: list of columns that contain categorical features
        encoded_category_lst (optional): list of names of the new features.
        If None, the name of the new features will be the one of the original
        feature followed by '_Churn'. Otherwise it should have the same length
        as the category_lst and contain the names of the new features.
        Defaults to None.

    Returns:
        df: pandas dataframe with new columns  containing the proportions of churned
    """
    logger.info("Turning categorical columns into proportion of churn")
    logger.info("Input categorical columns: %s", category_lst)
    encoded_category_lst = [
        f'{c}_Churn' for c in category_lst
    ] if encoded_category_lst is None else encoded_category_lst
    logger.info(
        "Output proportion of churn columns: %s",
        encoded_category_lst)
    assign_dict = {enc_c: df.groupby(c)['Churn'].transform(
        'mean') for c, enc_c in zip(category_lst, encoded_category_lst)}
    df = df.assign(**assign_dict)
    return df


def perform_feature_engineering(df):
    """Performs feature engineering.
    Creates columns with the proportion of churner for some relevant categorical variables

    The original function had a parameter response
    response: string of response name [optional argument that could be
    used for naming variables or index y column]
    -- I did not understand how that was intended to be used so I removed it.

    Args:
        df: pandas dataframe

    Returns:
        X_train: X training data
        X_test: X testing data
        y_train: y training data
        y_test: y testing data
    """
    logger.info("**** Feature engineering")
    logger.info("Encoding categorical variables")
    category_lst = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category']
    df = encoder_helper(df, category_lst)
    logger.info("Creating train test splits")
    y = df['Churn']
    X = df[const.KEEP_COLS]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test


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
    logger.info("Producing classification report for logistic regression")
    plt.rc('figure', figsize=(5, 5))
    fig = plt.figure()
    fig.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    fig.text(
        0.01, 0.05, str(
            classification_report(
                y_train, y_train_preds_lr)), {
            'fontsize': 10}, fontproperties='monospace')
    fig.text(0.01, 0.6, str('Logistic Regression Test'), {
             'fontsize': 10}, fontproperties='monospace')
    fig.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    save_into_folder(
        fig,
        'classification_report_logistic_regression',
        folder=const.RESULT_FOLDER)

    logger.info("Producing classification report for random forest")
    plt.rc('figure', figsize=(5, 5))
    fig = plt.figure()
    fig.text(0.01, 1.25, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    fig.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')
    fig.text(0.01, 0.6, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    fig.text(
        0.01, 0.7, str(
            classification_report(
                y_train, y_train_preds_rf)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    save_into_folder(
        fig,
        'classification_report_random_forest',
        folder=const.RESULT_FOLDER)


def roc_curve_plot(rfc, lrc, X_test, y_test):
    """Plots the oc curve and saves the output in the results folder.
    It also returns the figure.

    Args:
        rfc: random forest model
        lrc: logistic regression model
        X_test: test features
        y_test: test target

    Returns:
        fig: the roc plot
    """
    logger.info("Plotting roc curves for both models")
    lrc_plot = RocCurveDisplay.from_estimator(lrc, X_test, y_test)
    plt.close()
    fig = plt.figure(figsize=(15, 8))
    ax = plt.gca()
    RocCurveDisplay.from_estimator(rfc, X_test, y_test, ax=ax, alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.close()
    save_into_folder(
        fig, 'roc_curve_plot', folder=const.RESULT_FOLDER)
    return fig


def feature_importance_plot(model, X_data):
    """Creates the feature importance plot and saves it as .png in the images folder.

    Args:
        model: model object containing feature_importances_
        X_data: pandas dataframe of X values

    Returns:
        fig: the feature importance plot
    """
    logger.info("Plotting the feature importance")
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]
    # So they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    fig = plt.figure(figsize=(20, 5))
    ax = fig.gca()
    fig.suptitle("Feature Importance")
    ax.set_ylabel('Importance')
    ax.bar(range(X_data.shape[1]), importances[indices])
    ax.set_xticks(range(X_data.shape[1]), names, rotation=90)
    fig.tight_layout()
    plt.close()

    save_into_folder(
        fig, 'feature_importance_plot', folder=const.RESULT_FOLDER)
    return fig


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
    logger.info("**** Training the models")
    logger.info("Training logistic regression")
    lrc = LogisticRegression(solver='liblinear')
    lrc.fit(X_train, y_train)

    logger.info("Training cross validated random forest")
    rfc = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [200, 300],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 10],
        'criterion': ['gini', 'entropy'],
    }
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)
    rfc = cv_rfc.best_estimator_

    logger.info("**** Generating predictions")
    logger.info("Predicting using logistic regression")
    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)
    logger.info("Predicting using random forest")
    y_train_preds_rf = rfc.predict(X_train)
    y_test_preds_rf = rfc.predict(X_test)

    logger.info("**** Model evaluation")
    logger.info("Generating reports on model performances")
    classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf)
    roc_curve_plot(rfc, lrc, X_test, y_test)
    feature_importance_plot(rfc, pd.concat([X_train, X_test]))

    logger.info("**** Saving the models")
    logger.info("Saving logistic regression model as lrc_model.pkl")
    joblib.dump(lrc, os.path.join(const.MODEL_FOLDER, 'lrc_model.pkl'))
    logger.info("Saving random forest model as rfc_model.pkl")
    joblib.dump(rfc, os.path.join(const.MODEL_FOLDER, 'rfc_model.pkl'))


if __name__ == '__main__':
    bank_data_csv = os.path.join(const.DATA_FOLDER, "bank_data.csv")
    df = import_data(bank_data_csv)
    perform_eda(df)
    X_train_, X_test_, y_train_, y_test_ = perform_feature_engineering(df)
    train_models(X_train_, X_test_, y_train_, y_test_)
