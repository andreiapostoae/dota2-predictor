""" Module responsible for calculating advantages given a dataset of games """
import numpy as np

from tools.metadata import get_last_patch


def _update_dicts(game, synergy, counter):
    """ Updates the synergy and counter games given the game given as input
    Args:
        game: row of a mined pandas DataFrame
        synergy: synergy matrix
        counter: counter matrix
    Returns:
        None
    """
    radiant_win, radiant_heroes, dire_heroes = game[1], game[2], game[3]

    radiant_heroes = map(int, radiant_heroes.split(','))
    dire_heroes = map(int, dire_heroes.split(','))

    for i in range(5):
        for j in range(5):
            if i != j:
                synergy['games'][radiant_heroes[i] - 1, radiant_heroes[j] - 1] += 1
                synergy['games'][dire_heroes[i] - 1, dire_heroes[j] - 1] += 1

                if radiant_win:
                    synergy['wins'][radiant_heroes[i] - 1, radiant_heroes[j] - 1] += 1
                else:
                    synergy['wins'][dire_heroes[i] - 1, dire_heroes[j] - 1] += 1

            counter['games'][radiant_heroes[i] - 1, dire_heroes[j] - 1] += 1
            counter['games'][dire_heroes[i] - 1, radiant_heroes[j] - 1] += 1

            if radiant_win:
                counter['wins'][radiant_heroes[i] - 1, dire_heroes[j] - 1] += 1
            else:
                counter['wins'][dire_heroes[i] - 1, radiant_heroes[j] - 1] += 1


def _compute_winrates(synergy, counter, heroes_released):
    """ Calculates the winrate of every combination of heroes released from a synergy perspective
    and a counter perspective. The results are stored in the synergy and counter dictionaries
    using the 'winrate' key.
    Args:
        synergy: synergy matrix
        counter: counter matrix
        heroes_released: number of heroes released until current patch
    Returns:
        None
    """
    for i in range(heroes_released):
        for j in range(heroes_released):
            if i != j and i != 23 and j != 23:
                if synergy['games'][i, j] != 0:
                    synergy['winrate'][i, j] = synergy['wins'][i, j] / \
                                               float(synergy['games'][i, j])

                if counter['games'][i, j] != 0:
                    counter['winrate'][i, j] = counter['wins'][i, j] / \
                                               float(counter['games'][i, j])


def _adv_synergy(winrate_together, winrate_hero1, winrate_hero2):
    """ Given the winrate of 2 heroes played separately and together, return a score representing
    the advantage of the heroes being played together. There have been many tries in calculating
    this advantage score but simple winrate when playing together seems to work best.
    Args:
        winrate_together: winrate when both heroes are played in the same team
        winrate_hero1: general winrate of hero1
        winrate_hero2: general winrate of hero2
    Returns:
        advantage computed using the winrates given as input
    """
    return winrate_together


def _adv_counter(winrate_together, winrate_hero1, winrate_hero2):
    """ Given the winrate of one hero when playing against another hero and their separated
    winrates, return a score representing the advantage when hero1 is picked against hero2. There
    have been many tries in calculating this advantage score but simple winrate when playing against
    eachother seems to work best.
    Args:
        winrate_together: winrate when hero1 is picked against hero2
        winrate_hero1: general winrate of hero1
        winrate_hero2: general winrate of hero2
    Returns:
        advantage computed using the winrates given as input
    """
    return winrate_together


def _calculate_advantages(synergy, counter, heroes_released):
    """ Calculate base winrate for every hero and use it to compute advantages
    Args:
        synergy: synergy matrix
        counter: counter matrix
        heroes_released: number of heroes released in the current patch
    Returns:
        synergy matrix, counter_matrix using advantages
    """
    synergies = np.zeros((heroes_released, heroes_released))
    counters = np.zeros((heroes_released, heroes_released))

    base_winrate = np.zeros(heroes_released)

    for i in range(heroes_released):
        if i != 23:
            base_winrate[i] = np.sum(synergy['wins'][i]) / np.sum(synergy['games'][i])

    for i in range(heroes_released):
        for j in range(heroes_released):
            if i != j and i != 23 and j != 23:
                if synergy['games'][i, j] > 0:
                    synergies[i, j] = _adv_synergy(synergy['winrate'][i, j],
                                                   base_winrate[i],
                                                   base_winrate[j])
                else:
                    synergies[i, j] = 0

                if counter['games'][i, j] > 0:
                    counters[i, j] = _adv_counter(counter['winrate'][i, j],
                                                  base_winrate[i],
                                                  base_winrate[j])
                else:
                    counters[i, j] = 0

    return synergies, counters


def compute_advantages(dataset_df):
    """ Given a pandas DataFrame as input, calculate advantages and store them in synergy and
    counter dictionaries. The results are stored in files for easier later use.
    Args:
        dataset_df: pandas DataFrame containing the games to be analyzed
    Returns:
        synergy matrix and counter matrix using advantages
    """

    last_patch_info = get_last_patch()
    heroes_released = last_patch_info['heroes_released']

    synergy = dict()
    synergy['wins'] = np.zeros((heroes_released, heroes_released))
    synergy['games'] = np.zeros((heroes_released, heroes_released))
    synergy['winrate'] = np.zeros((heroes_released, heroes_released))

    counter = dict()
    counter['wins'] = np.zeros((heroes_released, heroes_released))
    counter['games'] = np.zeros((heroes_released, heroes_released))
    counter['winrate'] = np.zeros((heroes_released, heroes_released))

    dataset_np = dataset_df.values

    for row in dataset_np:
        _update_dicts(row, synergy, counter)

    _compute_winrates(synergy, counter, heroes_released)

    synergy_matrix, counter_matrix = _calculate_advantages(synergy, counter, heroes_released)

    # uncomment only for overwriting precomputed advantages - NOT RECOMMENDED
    # np.savetxt('pretrained/synergies_all.csv', synergy_matrix)
    # np.savetxt('pretrained/counters_all.csv', counter_matrix)

    return [synergy_matrix, counter_matrix]