# this module is responsible for solving heroes ID indices problem

import pandas as pd

COLUMNS = ['match_id', 'radiant_win', 'radiant_team', 'dire_team', 'avg_mmr', 'num_mmr',
           'game_mode', 'lobby_type']


def remapping_raw_data(data_csv_file_name, target_csv_file_name):
    """
    (1) remapping the heroes ID to indices from 1 - 116 (also missing ID 24)
    (2) make sure that game is 5V5
    (3) save new file as a .csv file
    :param data_csv_file_name: .csv file from raw OpenDota API
    :param target_csv_file_name: the name of file which is saved to hard disk
    :return: None
    """

    # remapping
    df = pd.read_csv(data_csv_file_name)
    df['radiant_team'] = map(lambda x: x.replace("119", "115"), df['radiant_team'])
    df['radiant_team'] = map(lambda x: x.replace("120", "116"), df['radiant_team'])
    df['dire_team'] = map(lambda x: x.replace("119", "115"), df['dire_team'])
    df['dire_team'] = map(lambda x: x.replace("120", "116"), df['dire_team'])

    # 5 V 5 filter
    df_condition = df.dire_team.map(lambda x: len(x.split(','))) < 5
    condition_list1 = df_condition[df_condition==True].index
    df_condition = df.radiant_team.map(lambda x: len(x.split(','))) < 5
    condition_list2 = df_condition[df_condition==True].index
    condition_list = condition_list1.append(condition_list2)
    df.drop(condition_list, inplace=True)

    # save
    df.to_csv(target_csv_file_name, index=False, columns=COLUMNS)


