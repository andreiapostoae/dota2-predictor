""" Module responsible with data augmentation """
import numpy as np


def augment_with_advantages(synergy, counter, radiant_heroes, dire_heroes):
    """ Computes a numpy array containing the features that have to be appended to a game
    Args:
        synergy: synergy matrix
        counter: counter matrix
        radiant_heroes: list of radiant heroes IDs
        dire_heroes: list of dire heroes IDs
    Returns:
        np.array containing synergy of radiant team, synergy of dire team and counter score of
        radiant against dire
    """
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
