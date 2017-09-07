from sklearn.externals import joblib
from os import listdir
import os
import numpy as np
from tools.metadata import get_hero_dict
from preprocessing.augmenter import augment_with_advantages
import operator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def query(mmr, radiant_heroes, dire_heroes, synergies, counters, similarities):
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

        print "Using closest model available: %d MMR model" % final_mmr

        model_dict = joblib.load(os.path.join("pretrained", str(final_mmr) + ".pkl"))

    scaler = model_dict['scaler']
    model = model_dict['model']

    if len(radiant_heroes) + len(dire_heroes) == 10:
        features = np.zeros(231)
        for i in range(5):
            features[radiant_heroes[i] - 1] = 1
            features[dire_heroes[i] - 1 + 114] = 1

        extra_data = augment_with_advantages(synergies, counters, radiant_heroes, dire_heroes)
        features[-3:] = extra_data

        features_reshaped = features.reshape(1, -1)
        features_final = scaler.transform(features_reshaped)

        probability = model.predict_proba(features_final)[:, 1] * 100

        logger.info("Radiant chance to win: %.3f%%", probability)
        logger.info("Dire chance to win: %.3f%%", (100 - probability))

    else:
        all_heroes = radiant_heroes + dire_heroes
        base_similarity_radiant = 0
        base_similarity_dire = 0

        radiant = len(radiant_heroes) == 4

        for i in range(4):
            for j in range(4):
                if i > j:
                    base_similarity_radiant += similarities[radiant_heroes[i], radiant_heroes[j]]
                    base_similarity_dire += similarities[dire_heroes[i], dire_heroes[j]]

        query_base = np.zeros((114, 231))

        for i in range(114):
            if radiant:
                radiant_heroes.append(i + 1)
            else:
                dire_heroes.append(i + 1)

            for j in range(5):
                query_base[i][radiant_heroes[j] - 1] = 1
                query_base[i][dire_heroes[j] - 1 + 114] = 1

            query_base[i][-3:] = augment_with_advantages(synergies, counters, radiant_heroes, dire_heroes)

            if radiant:
                del radiant_heroes[-1]
            else:
                del dire_heroes[-1]

        np.set_printoptions(suppress=True)

        if radiant:
            probs = model.predict_proba(scaler.transform(query_base))[:, 1]
        else:
            probs = model.predict_proba(scaler.transform(query_base))[:, 0]

        heroes_dict = get_hero_dict()
        similarities_list = []

        results_dict = {}
        for i, prob in enumerate(probs):
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

        new_list = [x for x in results_list if x[1][1] < max_similarity_allowed]

        for element in new_list:
            print element
