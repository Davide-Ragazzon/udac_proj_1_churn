# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
Project for the Udacity course.
The aim of this part of the course is to learn about good practices for writing clean code.
This data science project predicts churn in a bank.

### Inputs and outputs



- Folders used for inputs and outputs can be specified in `constants.py`:
    - `DATA_FOLDER`: (default `./data`) raw data 
    - `IMG_FOLDER`: (default `./images`) exploratory data analysis (EDA) plots
    - `MODEL_FOLDER`: (default `./models`) pickled models
    - `RESULT_FOLDER`: (default `./results`) model reports, feature importance and ROC curves
    - `LOG_FOLDER`: (default `./logs`) logs
- Raw data are provided as the .csv file `bank_data.csv` in the `DATA_FOLDER`
- `constants.py` also allows to specify:
    - `KEEP_COLS`: features used for modeling
    - `RESULTS_LOG`:  (default `./logs/churn_library.log`) the file where the progress is logged (intended to be located within the `LOG_FOLDER`)
    - `TMP_TEST_FOLDER`: (default `./tmp`) folder to be used for selected tests involving file creation

### Analysis
The following data science analysis is performed:
- Data are loaded from the `DATA_FOLDER` 
- EDA is performed and the resulting plots are saved in the `IMG_FOLDER`
- Features are engineered, including encoding categorical columns into proportion of churned in that category
- Data are split into train and test set
- Cross-validated random forest and a logistic regression are trained and saved in the `MODEL_FOLDER`
- Predictions are generated
- Model performance is evaluated and reports are saved in the `RESULT_FOLDER`

A log of the progress of this analysis can be found in the `RESULT_LOG` file

## Running Files

The analysis described aboved are peroformed by running  
```python churn_library.py```

Unit tests for all the functions in `churn_library.py` are performed by running  
```python churn_script_logging_and_tests.py```




