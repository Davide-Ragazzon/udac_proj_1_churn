"""
Module for testing the functions in churn_library.
"""

import pandas as pd
import logging
import os

import matplotlib.pyplot as plt

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


def test_classification_report_image(classification_report_image):
    pass


def test_roc_curve_plot():
    pass


def test_feature_importance_plot(feature_importance_plot):
    pass


def test_train_models(train_models):
    '''
    test train_models
    '''


if __name__ == "__main__":
    logger.info("**** Performing tests for functions in churn_library.py")
    # test_save_into_folder(cl.save_into_folder)
    # test_perform_eda(cl.perform_eda)
    test_encoder_helper(cl.encoder_helper)
