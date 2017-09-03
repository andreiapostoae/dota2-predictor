import numpy as np
from tools.metadata import get_last_patch


def _update_dicts(game, synergy, counter):
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
    # exclude unused index 23
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
    # return ((winrate_together - winrate_hero1) / winrate_hero1) + \
    #        ((winrate_together - winrate_hero2) / winrate_hero2)
    return winrate_together


def _adv_counter(winrate_together, winrate_hero1, winrate_hero2):
    # return ((winrate_together - winrate_hero1) / winrate_hero1) - \
    #        ((1 - winrate_together - winrate_hero2) / winrate_hero2)
    return winrate_together


def _calculate_advantages(synergy, counter, heroes_released):
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


def load_advantages(file_name):
    synergy_matrix = np.loadtxt('synergy_' + file_name)
    counter_matrix = np.loadtxt('counter_' + file_name)

    return [synergy_matrix, counter_matrix]


def compute_advantages(dataset_df):
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

    np.savetxt('synergies_all.csv', synergy_matrix)
    np.savetxt('counters_all.csv', counter_matrix)

    return [synergy_matrix, counter_matrix]