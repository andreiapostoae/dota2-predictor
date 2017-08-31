from tools.metadata import get_last_patch
# from joblib import Parallel, delayed
import multiprocessing
import numpy as np


def augment_with_advantages(synergy, counter, radiant_heroes, dire_heroes):
    synergy_radiant = 0
    synergy_dire = 0
    counter_score = 0

    for i in range(5):
        for j in range(5):
            if i > j:
                synergy_radiant += synergy[radiant_heroes[i] - 1][radiant_heroes[j] - 1]
                synergy_dire += synergy[dire_heroes[i] - 1][dire_heroes[j] - 1]

            counter_score += counter[radiant_heroes[i] - 1][dire_heroes[j] - 1]

    return np.array([synergy_radiant, synergy_dire, counter_score])