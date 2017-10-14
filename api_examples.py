""" Module for showing examples of dota predictor's API usage """
import logging
import numpy as np

from preprocessing.dataset import read_dataset
from tools.metadata import get_last_patch, get_patch
from tools.miner import mine_data
from training.cross_validation import evaluate
from training.query import query
from visualizing.dataset_stats import pick_statistics, winrate_statistics, mmr_distribution
from visualizing.hero_combinations import plot_synergies, plot_counters
from visualizing.hero_map import plot_hero_map
from visualizing.learning_curve import plot_learning_curve


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def mine_data_example():
    # mine 100 games from the last patch
    mined_df = mine_data(stop_at=100)
    logger.info("First 5 rows from the mined dataframe are: \n%s", mined_df.head().to_string())

    # mine 1000 games between given match IDs and save file
    mine_data(file_name='mine_test.csv',
              first_match_id=3492535023,
              last_match_id=3498023575,
              stop_at=1000)


def patch_data_example():
    # information about latest patch released
    last_patch = get_last_patch()
    logger.info("Latest patch released is: %s", last_patch['patch_name'])
    logger.info("Match IDs for latest patch span between %s and %s", last_patch['first_match_id'],
                last_patch['last_match_id'])

    # information about specific patch
    patch = get_patch('7.06e')
    logger.info("Match IDs for %s span between %d and %d", patch['patch_name'],
                patch['first_match_id'], patch['last_match_id'])


def load_dataset_example():
    # load dataset from csv using precomputed advantages from the entire train dataset
    dataset_simple, advantages = read_dataset('706e_train_dataset.csv', low_mmr=4500)
    logger.info("The features have shape: %s", dataset_simple[0].shape)
    logger.info("The labels have shape: %s", dataset_simple[1].shape)
    logger.info("Synergies for Anti-Mage (index 1): \n%s", advantages[0][0, :])
    logger.info("Counters for Monkey King (index 114): \n%s", advantages[1][113, :])

    # load dataset from csv and recompute advantages for this specific dataset
    dataset_advanced, advantages_computed = read_dataset('706e_train_dataset.csv',
                                                         low_mmr=2000,
                                                         high_mmr=2500,
                                                         advantages=True)
    logger.info("The features have shape: %s", dataset_advanced[0].shape)
    logger.info("The labels have shape: %s", dataset_advanced[1].shape)
    logger.info("Synergies for Anti-Mage (index 1): \n%s", advantages_computed[0][0, :])
    logger.info("Counters for Monkey King (index 114): \n%s", advantages_computed[1][113, :])


def training_example():
    dataset_train, _ = read_dataset('706e_train_dataset.csv', low_mmr=4500)
    dataset_test, _ = read_dataset('706e_test_dataset.csv', low_mmr=4500)

    # cv is the number of folds to be used when cross validating (default is 5)
    # save_model is the path where the model should be saved (default None)
    evaluate(dataset_train, dataset_test, cv=7, save_model='test.pkl')


def query_example():
    # having the models pretrained on specific MMR ranges (see pretrained folder), query the result
    # of a game by finding the closest model

    # query for the result given the 5v5 configuration in a game around 3000 average MMR
    # radiant team: Huskar, Clinkz, Lifestealer, Luna, Lich
    # dire team: Venomancer, Faceless Void, Leshrac, Ancient Apparition, Broodmother
    full_result = query(3000,
                        [59, 56, 54, 48, 31],
                        [40, 41, 52, 68, 61])
    logger.info("The result of the full query is: %s", full_result)

    # query for the result given the 4v5 or 5v4 configuration in a game around 3000 average MMR
    # radiant team: Huskar, Clinkz, Lifestealer, Luna, Lich
    # dire team: Venomancer, Faceless Void, Leshrac, Ancient Apparition
    # the missing element of the 2nd list is the one the suggestion is made for
    # the result is a list of (hero, (win_chance, overall_team_similarity)) sorted by win_chance
    partial_result = query(3000,
                           [59, 56, 54, 48, 31],
                           [40, 41, 52, 68])
    logger.info("The result of the partial query is: \n%s", partial_result)


def visualize_data_example():
    # in order to use plotly (for most of these examples), you need to create an account and
    # configure your credentials; the plots will be saved to your online account
    # see https://plot.ly/python/getting-started/

    # plot learning curve for a loaded dataset with either matplotlib or plotly
    # subsets represents the number of points where the accuracies are evaluated
    # cv represents the number of folds for each point of the evaluation
    features, _ = read_dataset('706e_train_dataset.csv', low_mmr=3000, high_mmr=3500)
    plot_learning_curve(features[0], features[1], subsets=20, cv=3, mmr=3250, tool='matplotlib')

    # the rest of the plots were implemented only for plotly because of their size

    # plot win rate statistics
    winrate_statistics(features, '3000 - 3500')

    # plot pick rate statistics
    pick_statistics(features, '3000 - 3500')

    # plot mmr distribution
    mmr_distribution('706e_train_dataset.csv')

    # plot synergies and counters for hero combinations
    # they are loaded from the pretrained folder
    plot_synergies()
    plot_counters()

    # plot hero map containing the heroes grouped by the similarity of their role
    # the heroes are clustered by roles: support, offlane, mid, carry
    plot_hero_map('706e_train_dataset.csv')


def main():
    mine_data_example()
    patch_data_example()
    load_dataset_example()
    training_example()
    query_example()
    visualize_data_example()


if __name__ == '__main__':
    main()
