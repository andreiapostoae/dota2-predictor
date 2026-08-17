""" Module responsible with training, evaluating and cross validation of data """
import logging
import numpy as np

import joblib
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate(train_data, test_data, cv=5, save_model=None):
    """ Given train data, performs cross validation using cv folds, then calculates the score on
    test data. The metric used is roc_auc_score from sklearn. Before training, the data is
    normalized and the scaler used is saved for scaling the test data in the same manner.

    Args:
        train_data: list containing x_train and y_train
        test_data: list containing x_test and y_test
        cv: number of folds to use in cross validation
        save_model: if given, the model is saved to this path
    Returns:
        train size, test size, roc_auc score
    """
    x_train, y_train = train_data
    x_test, y_test = test_data

    scaler = StandardScaler()
    scaler.fit(x_train)

    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    cross_val_mean = -1
    if cv > 0:
        model = LogisticRegression(C=0.005, random_state=42)
        cross_val_scores = cross_val_score(model, x_train, y_train, cv=cv, scoring='roc_auc',
                                           n_jobs=-1)

        cross_val_mean = np.mean(cross_val_scores)
        logger.info("Cross validation scores over the training set (%d folds): %.3f +/- %.3f", cv,
                    cross_val_mean,
                    np.std(cross_val_scores))

    model = LogisticRegression(C=0.005, random_state=42)
    model.fit(x_train, y_train)

    probabilities = model.predict_proba(x_test)
    roc_auc = roc_auc_score(y_test, probabilities[:, 1])

    labels = model.predict(x_test)
    acc_score = accuracy_score(y_test, labels)

    if save_model:
        model_dict = {}
        model_dict['scaler'] = scaler
        model_dict['model'] = model

        joblib.dump(model_dict, save_model)

    logger.info("Test ROC AUC: %.3f", roc_auc)
    logger.info("Test accuracy score: %.3f", acc_score)

    return (x_train.shape[0], x_test.shape[0], cross_val_mean, roc_auc, acc_score)
