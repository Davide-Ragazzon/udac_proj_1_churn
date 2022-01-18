# library doc string


from sklearn.metrics import RocCurveDisplay
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import logging

import constants as const
import churn_utils as cu

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import plot_roc_curve, classification_report

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
    cu.save_into_folder(fig, 'eda_churn', logger)

    logger.info("Plotting distribution of Customer_Age")
    fig = plt.figure(figsize=(20, 10))
    ax = fig.gca()
    df['Customer_Age'].hist(ax=ax)
    cu.save_into_folder(fig, 'eda_age', logger)

    logger.info("Plotting distribution of Marital_Status")
    fig = plt.figure(figsize=(20, 10))
    ax = fig.gca()
    df.Marital_Status.value_counts('normalize').plot(kind='bar', ax=ax)
    cu.save_into_folder(fig, 'eda_marital_status', logger)

    logger.info("Plotting distribution of Total_Trans_Ct")
    fig = plt.figure(figsize=(20, 10))
    ax = fig.gca()
    sns.distplot(df['Total_Trans_Ct'], ax=ax)
    cu.save_into_folder(fig, 'eda_tot_trans_ct', logger)

    logger.info("Plotting correlation between features")
    fig = plt.figure(figsize=(20, 10))
    ax = fig.gca()
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2, ax=ax)
    cu.save_into_folder(fig, 'eda_corr', logger)


def encoder_helper(df, category_lst, encoded_category_lst=None):
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
    logger.info("Turning categorical columns into proportion of churn")
    logger.info(f"Input categorical columns: {category_lst}")
    encoded_category_lst = [
        f'{c}_Churn' for c in category_lst] if encoded_category_lst is None else encoded_category_lst
    logger.info(f"Output proportion of churn columns: {encoded_category_lst}")
    assign_dict = {enc_c: df.groupby(c)['Churn'].transform(
        'mean') for c, enc_c in zip(category_lst, encoded_category_lst)}
    df = df.assign(**assign_dict)
    return df


def perform_feature_engineering(df):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column] -- I do not understand how this is intended to be used

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    category_lst = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category']
    df = encoder_helper(df, category_lst)
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
    cu.save_into_folder(
        fig,
        'classification_report_logistic_regression',
        logger,
        folder=const.RESULTS_FOLDER)

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
    cu.save_into_folder(
        fig,
        'classification_report_random_forest',
        logger,
        folder=const.RESULTS_FOLDER)


def roc_curve_plot(rfc, lrc, X_test, y_test):
    logger.info("Plotting roc curves for both models")
    lrc_plot = RocCurveDisplay.from_estimator(lrc, X_test, y_test)
    plt.close()
    fig = plt.figure(figsize=(15, 8))
    ax = plt.gca()
    rfc_disp = RocCurveDisplay.from_estimator(
        rfc, X_test, y_test, ax=ax, alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.close()
    cu.save_into_folder(
        fig, 'roc_curve_plot', logger, folder=const.RESULTS_FOLDER)
    return fig


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
    logger.info("Training logistic regression")
    lrc = LogisticRegression(solver='liblinear')
    lrc.fit(X_train, y_train)

    logger.info("Training cross validated random forest")
    rfc = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [200, 300],
        # 'max_features': ['auto', 'sqrt'],  # TODO: Uncomment parameters when the code is ready
        'max_depth': [4, 5, 10],
        # 'criterion': ['gini', 'entropy'],  # TODO: Uncomment parameters when the code is ready
    }
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)
    rfc = cv_rfc.best_estimator_

    logger.info("Generating reports on model performances")
    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)
    y_train_preds_rf = rfc.predict(X_train)
    y_test_preds_rf = rfc.predict(X_test)

    classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf)
    roc_curve_plot(rfc, lrc, X_test, y_test)


if __name__ == '__main__':
    bank_data_csv = os.path.join(const.DATA_FOLDER, "bank_data.csv")
    df = import_data(bank_data_csv)
    perform_eda(df)
    X_train, X_test, y_train, y_test = perform_feature_engineering(df)
    train_models(X_train, X_test, y_train, y_test)
