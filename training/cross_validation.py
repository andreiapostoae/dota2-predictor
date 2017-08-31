from sklearn.model_selection import cross_val_score
from preprocessing.augmenter import augment_with_advantages


def cv_score(feature_map,
             estimator,
             advantages=None,
             cv=5):
    x_train = feature_map[:, :-1]
    y_train = feature_map[:, -1]

    if advantages:
        augment_with_advantages(x_train, advantages)