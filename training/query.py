""" Module responsible for querying the result of a game """

import operator
import os
import logging
import numpy as np

from os import listdir
from sklearn.externals import joblib

from preprocessing.augmenter import augment_with_advantages
from tools.metadata import get_hero_dict, get_last_patch


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _query_missing(model,
                   scaler,
                   radiant_heroes,
                   dire_heroes,
                   synergies,
                   counters,
                   similarities,
                   heroes_released):
    """ Query the best missing hero that can be picked given 4 heroes in one
    team and 5 heroes in the other.

    Args:
        model: estimator that has fitted the data
        scaler: the scaler used for fitting the data
        radiant_heroes: list of hero IDs from radiant team
        dire_heroes: list of hero IDs from dire team
        synergies: matrix defining the synergy scores between heroes
        counters: matrix defining the counter scores between heroes
        similarities: matrix defining similarities between heroes
        heroes_released: number of heroes released in the queried patch
    Returns:
        list of variable length containing hero suggestions
    """
    all_heroes = radiant_heroes + dire_heroes
    base_similarity_radiant = 0
    base_similarity_dire = 0

    radiant = len(radiant_heroes) == 4

    for i in range(4):
        for j in range(4):
            if i > j:
                base_similarity_radiant += similarities[radiant_heroes[i], radiant_heroes[j]]
                base_similarity_dire += similarities[dire_heroes[i], dire_heroes[j]]

    query_base = np.zeros((heroes_released, 2 * heroes_released + 3))

    for i in range(heroes_released):
        if radiant:
            radiant_heroes.append(i + 1)
        else:
            dire_heroes.append(i + 1)

        for j in range(5):
            query_base[i][radiant_heroes[j] - 1] = 1
            query_base[i][dire_heroes[j] - 1 + heroes_released] = 1

        query_base[i][-3:] = augment_with_advantages(synergies,
                                                     counters,
                                                     radiant_heroes,
                                                     dire_heroes)

        if radiant:
            del radiant_heroes[-1]
        else:
            del dire_heroes[-1]

    if radiant:
        probabilities = model.predict_proba(scaler.transform(query_base))[:, 1]
    else:
        probabilities = model.predict_proba(scaler.transform(query_base))[:, 0]

    heroes_dict = get_hero_dict()
    similarities_list = []

    results_dict = {}
    for i, prob in enumerate(probabilities):
        if i + 1 not in all_heroes and i != 23:
            if radiant:
                similarity_new = base_similarity_radiant
                for j in range(4):
                    similarity_new += similarities[i + 1][radiant_heroes[j]]
                similarities_list.append(similarity_new)
            else:
                similarity_new = base_similarity_dire
                for j in range(4):
                    similarity_new += similarities[i + 1][dire_heroes[j]]
                similarities_list.append(similarity_new)

            results_dict[heroes_dict[i + 1]] = (prob, similarity_new)

    results_list = sorted(results_dict.items(), key=operator.itemgetter(1), reverse=True)

    similarities_list.sort()

    max_similarity_allowed = similarities_list[len(similarities_list) / 4]

    filtered_list = [x for x in results_list if x[1][1] < max_similarity_allowed]

    return filtered_list


def _query_full(model,
                scaler,
                radiant_heroes,
                dire_heroes,
                synergies,
                counters,
                heroes_released):
    """ Query the result of a game when both teams have their line-ups
    finished.

    Args:
        model: estimator that has fitted the data
        scaler: the scaler used for fitting the data
        radiant_heroes: list of hero IDs from radiant team
        dire_heroes: list of hero IDs from dire team
        synergies: matrix defining the synergy scores between heroes
        counters: matrix defining the counter scores between heroes
        heroes_released: number of heroes released in the queried patch
    Returns:
        string with info about the predicted winner team
    """
    features = np.zeros(2 * heroes_released + 3)
    for i in range(5):
        features[radiant_heroes[i] - 1] = 1
        features[dire_heroes[i] - 1 + heroes_released] = 1

    extra_data = augment_with_advantages(synergies, counters, radiant_heroes, dire_heroes)
    features[-3:] = extra_data

    features_reshaped = features.reshape(1, -1)
    features_final = scaler.transform(features_reshaped)

    probability = model.predict_proba(features_final)[:, 1] * 100

    if probability > 50:
        return "Radiant has %.3f%% chance" % probability
    else:
        return "Dire has %.3f%% chance" % (100 - probability)


def query(mmr, 
          radiant_heroes, 
          dire_heroes, 
          synergies=None, 
          counters=None, 
          similarities=None):
    if similarities is None:
        sims = np.loadtxt('pretrained/similarities_all.csv')
    else:
        sims = np.loadtxt(similarities)

    if counters is None:        
        cnts = np.loadtxt('pretrained/counters_all.csv')
    else:
        cnts = np.loadtxt(counters)

    if synergies is None:
        syns = np.loadtxt('pretrained/synergies_all.csv')
    else:
        syns = np.loadtxt(synergies)

    if mmr < 0 or mmr > 10000:
        logger.error("MMR should be a number between 0 and 10000")
        return

    if mmr < 2000:
        model_dict = joblib.load(os.path.join("pretrained", "2000-.pkl"))
        logger.info("Using 0-2000 MMR model")
    elif mmr > 5000:
        model_dict = joblib.load(os.path.join("pretrained", "5000+.pkl"))
        logger.info("Using 5000-10000 MMR model")
    else:
        file_list = [int(valid_file[:4]) for valid_file in listdir('pretrained')
                     if '.pkl' in valid_file]

        file_list.sort()

        min_distance = 10000
        final_mmr = -1000

        for model_mmr in file_list:
            if abs(mmr - model_mmr) < min_distance:
                min_distance = abs(mmr - model_mmr)
                final_mmr = model_mmr

        logger.info("Using closest model available: %d MMR model", final_mmr)

        model_dict = joblib.load(os.path.join("pretrained", str(final_mmr) + ".pkl"))

    scaler = model_dict['scaler']
    model = model_dict['model']

    last_patch_info = get_last_patch()
    heroes_released = last_patch_info['heroes_released']

    if len(radiant_heroes) + len(dire_heroes) == 10:
        return _query_full(model,
                           scaler,
                           radiant_heroes,
                           dire_heroes,
                           syns,
                           cnts,
                           heroes_released)

    return _query_missing(model,
                          scaler,
                          radiant_heroes,
                          dire_heroes,
                          syns,
                          cnts,
                          sims,
                          heroes_released)
