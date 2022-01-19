"""
Module for testing the functions in churn_library.
"""

import logging
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import churn_library as cl
import constants as const

#import churn_library_solution as cls


logging.basicConfig(
    filename=const.RESULTS_LOG,
    level=logging.INFO,
    filemode='w',
    #format='%(name)s - %(levelname)s - %(message)s',
    format='%(asctime)s %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

logger = logging.getLogger(os.path.basename(__file__))


def test_import_data(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = import_data("./data/bank_data.csv")
        logger.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logger.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logger.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_save_into_folder(save_into_folder):
    # Start with a clean folder
    folder = const.TMP_TEST_FOLDER
    expected_file = os.path.join(folder, 'tmp_test.png')
    if os.path.isfile(expected_file):
        os.remove(expected_file)
    # Prepare figure to be saved
    fig = plt.figure()
    ax = fig.gca()
    ax.plot([1, 3, 2])
    save_into_folder(fig, 'tmp_test', folder=folder)
    # Check that the figure was saved
    try:
        assert os.path.isfile(expected_file)
        logger.info('Testing save_into_folder: SUCCESS')
    except AssertionError:
        logger.error('Testing save_into_folder: expected file not found')


def test_perform_eda(perform_eda):
    '''
    test perform eda function
    Checks if files exist as suggested in the Udacity questions
    https://knowledge.udacity.com/questions/602346
    '''
    df = pd.DataFrame({
        'Churn': [0, 1, 0, 1],
        'Customer_Age': [20, 30, 40, 50],
        'Marital_Status': ['Married', 'Single', 'Married', 'Unknown'],
        'Total_Trans_Ct': [55, 65, 75, 85],
    })
    perform_eda(df)

    try:
        assert os.path.isfile(os.path.join(const.IMG_FOLDER, 'eda_churn.png'))
        assert os.path.isfile(os.path.join(const.IMG_FOLDER, 'eda_age.png'))
        assert os.path.isfile(
            os.path.join(
                const.IMG_FOLDER,
                'eda_marital_status.png'))
        assert os.path.isfile(
            os.path.join(
                const.IMG_FOLDER,
                'eda_tot_trans_ct.png'))
        assert os.path.isfile(os.path.join(const.IMG_FOLDER, 'eda_corr.png'))
        logger.info('Testing perform_eda: SUCCESS')
    except AssertionError:
        logger.error('Testing perform_eda: expected file not found')


def test_encoder_helper(encoder_helper):
    '''
    test encoder helper
    '''
    df = pd.DataFrame({
        'Churn': [0, 1, 1, 1],
        'Marital_Status': ['Married', 'Single', 'Married', 'Unknown'],
    })
    df = encoder_helper(df, ['Marital_Status'])

    try:
        assert 'Marital_Status_Churn' in df.columns
        logger.info('Testing encoder_helper: SUCCESS - Expected column found')
    except AssertionError:
        logger.error('Testing encoder_helper: expected column not found')

    if 'Marital_Status_Churn' not in df.columns:
        logger.info(
            'Testing encoder_helper: Skipping value test because of missing requred field')
        return
    try:
        assert df['Marital_Status_Churn'].tolist() == [0.5, 1.0, 0.5, 1.0]
        logger.info('Testing encoder_helper: SUCCESS - Expected values found')
    except AssertionError:
        logger.error('Testing encoder_helper: Expected values not found')


def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''
    df = pd.DataFrame({
        'Churn': [0, 1, 0, 1],
        'Customer_Age': [20, 30, 40, 50],
        'Marital_Status': ['Married', 'Single', 'Married', 'Unknown'],
        'Total_Trans_Ct': [55, 65, 75, 85],
        'Gender': ['M', 'F', 'M', 'F'],
        'Education_Level': ['Uneducated', 'Uneducated', 'High school', 'Graduate'],
        'Income_Category': ['Less than $40K', '80k-120k', '60k-80k', '60k-80k'],
        'Card_Category': ['Blue', 'Blue', 'Blue', 'Blue', ],
    })
    other_needed_cols = [c for c in const.KEEP_COLS if c not in df.columns]
    # Filled with dummy values
    df = df.assign(**{c: [1, 1, 1, 1] for c in other_needed_cols})
    result = perform_feature_engineering(df)

    try:
        assert len(result) == 4
        logger.info(
            'Testing perform_feature_engineering: SUCCESS - Correct number of outputs')
    except AssertionError:
        logger.error(
            'Testing perform_feature_engineering: Wrong number of outputs')

    X_train = result[0]
    try:
        assert X_train.shape == (2, 19)
        logger.info(
            'Testing perform_feature_engineering: SUCCESS - Expected shape of train set')
    except AssertionError:
        logger.error(
            'Testing perform_feature_engineering: Wrong shape of train set')


def test_classification_report_image(classification_report_image):
    y_train = np.array([1, 0, 0, 0, 1, 1, 1])
    y_test = np.array([0, 0, 1, 1, 1])
    y_train_preds_lr = np.array([1, 0, 1, 1, 0, 0, 1])
    y_train_preds_rf = np.array([1, 0, 0, 1, 0, 1, 1])
    y_test_preds_lr = np.array([0, 0, 1, 0, 0])
    y_test_preds_rf = np.array([0, 0, 1, 1, 0])
    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)
    try:
        assert os.path.isfile(
            os.path.join(
                const.RESULT_FOLDER,
                'classification_report_logistic_regression.png'))
        assert os.path.isfile(
            os.path.join(
                const.RESULT_FOLDER,
                'classification_report_random_forest.png'))
        logger.info('Testing classification_report_image: SUCCESS')
    except AssertionError:
        logger.error(
            'Testing classification_report_image: expected file not found')


def test_roc_curve_plot(roc_curve_plot):
    y = np.array([0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0])
    noise = np.array([1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1]) * 0.1
    var_1 = y + noise
    X = pd.DataFrame({'var_1': var_1})
    lrc = LogisticRegression()
    lrc.fit(X, y)
    rfc = lrc  # Dummy for the test
    fig = roc_curve_plot(rfc, lrc, X, y)

    try:
        matplotlib_fig = plt.figure()
        if not isinstance(fig, type(matplotlib_fig)):
            raise TypeError
        logger.info(
            'Testing roc_curve_plot: SUCCESS - output is a matplotlib figure')
    except TypeError:
        logger.error(
            'Testing roc_curve_plot: output is not a matplotlib figure')
    try:
        expected_file = os.path.join(const.RESULT_FOLDER, 'roc_curve_plot.png')
        assert os.path.isfile(expected_file)
        logger.info('Testing roc_curve_plot: SUCCESS - expected file found')
    except AssertionError:
        logger.error(
            'Testing roc_curve_plot: expected file not found')


def test_feature_importance_plot(feature_importance_plot):
    y = np.array([0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0])
    noise = np.array([1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1]) * 0.1
    var_1 = y + noise
    var_2 = np.square(noise)
    X = pd.DataFrame({'var_1': var_1, 'var_2': var_2})
    rfc = RandomForestClassifier()
    rfc.fit(X, y)
    fig = feature_importance_plot(rfc, X)

    try:
        matplotlib_fig = plt.figure()
        if not isinstance(fig, type(matplotlib_fig)):
            raise TypeError
        logger.info(
            'Testing feature_importance_plot: SUCCESS - output is a matplotlib figure')
    except TypeError:
        logger.error(
            'Testing feature_importance_plot: output is not a matplotlib figure')
    try:
        expected_file = os.path.join(
            const.RESULT_FOLDER,
            'feature_importance_plot.png')
        assert os.path.isfile(expected_file)
        logger.info(
            'Testing feature_importance_plot: SUCCESS - expected file found')
    except AssertionError:
        logger.error(
            'Testing feature_importance_plot: expected file not found')


def test_train_models(train_models):
    '''
    test train_models
    '''
    y = np.array([0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0])
    noise = np.array([1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1]) * 0.1
    var_1 = y + noise
    var_2 = np.square(noise)
    X = pd.DataFrame({'var_1': var_1, 'var_2': var_2})
    train_models(X, X, y, y)
    files_ok = None
    try:
        assert os.path.isfile(
            os.path.join(
                const.MODEL_FOLDER,
                'lrc_model.pkl'))
        assert os.path.isfile(
            os.path.join(
                const.MODEL_FOLDER,
                'rfc_model.pkl'))
        logger.info('Testing train_models: SUCCESS')
        files_ok = True
    except AssertionError:
        logger.error('Testing train_models: expected file not found')
        files_ok = False

    if not files_ok:
        logger.info(
            'Testing train_models: Skipping type check because of missing requred file')
        return
    lrc = joblib.load(os.path.join(const.MODEL_FOLDER, 'lrc_model.pkl'))
    rfc = joblib.load(os.path.join(const.MODEL_FOLDER, 'rfc_model.pkl'))

    try:
        if not isinstance(lrc, type(LogisticRegression())):
            raise TypeError
        logger.info(
            'Testing train_models: SUCCESS - correct type for logistic regression model')
    except TypeError:
        logger.error(
            'Testing train_models: wrong type for logistic regression model')
    try:
        if not isinstance(rfc, type(RandomForestClassifier())):
            raise TypeError
        logger.info(
            'Testing train_models: SUCCESS - correct type for random forest model')
    except TypeError:
        logger.error(
            'Testing train_models: wrong type for random forest model')


if __name__ == "__main__":
    logger.info(
        "**** Performing tests for all the functions in churn_library.py")
    test_save_into_folder(cl.save_into_folder)
    test_perform_eda(cl.perform_eda)
    test_encoder_helper(cl.encoder_helper)
    test_perform_feature_engineering(cl.perform_feature_engineering)
    test_classification_report_image(cl.classification_report_image)
    test_roc_curve_plot(cl.roc_curve_plot)
    test_feature_importance_plot(cl.feature_importance_plot)
    test_train_models(cl.train_models)
