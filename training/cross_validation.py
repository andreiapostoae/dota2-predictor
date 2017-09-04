from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.externals import joblib
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate(train_data, test_data, cv=5, save_model=None):
    x_train, y_train = train_data
    x_test, y_test = test_data

    scaler = StandardScaler()
    scaler.fit(x_train)

    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    if cv > 0:
        model = LogisticRegression(C=0.005, random_state=42)
        cross_val_scores = cross_val_score(model, x_train, y_train, cv=cv, scoring='roc_auc')
        logger.info("Cross validation scores over the training set (%d folds): %.3f +/- %.3f", cv,
                    np.mean(cross_val_scores),
                    np.std(cross_val_scores))

    model = LogisticRegression(C=0.005, random_state=42)
    model.fit(x_train, y_train)

    probabilities = model.predict_proba(x_test)
    accuracy = roc_auc_score(y_test, probabilities[:, 1])

    if save_model:
        model_dict = {}
        model_dict['scaler'] = scaler
        model_dict['model'] = model

        joblib.dump(model_dict, save_model)

    logger.info("Test accuracy: %.3f", accuracy)

    return (x_train.shape[0], x_test.shape[0], accuracy)
