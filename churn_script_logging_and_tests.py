"""
Module for testing the functions in churn_library.
"""

import os
import logging
#import churn_library_solution as cls

import constants as const
import churn_library as cl

import matplotlib.pyplot as plt

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
        print('yes')
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
    '''


def test_encoder_helper(encoder_helper):
    '''
    test encoder helper
    '''


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
    test_save_into_folder(cl.save_into_folder)
