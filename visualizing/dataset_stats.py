import numpy as np
from tools.metadata import get_hero_dict
import operator
import pandas as pd

import plotly.graph_objs as go
import plotly.plotly as py


def winrate_statistics(dataset_df, mmr_info):
    x_data, y_data = dataset_df

    wins = np.zeros(114)
    games = np.zeros(114)
    winrate = np.zeros(114)

    for idx, game in enumerate(x_data):
        for i in range(228):
            if game[i] == 1:
                games[i % 114] += 1

                if y_data[idx] == 1:
                    if i < 114:
                        wins[i] += 1
                else:
                    if i >= 114:
                        wins[i - 114] += 1

    winrate = wins / games

    winrate_dict = dict()
    hero_dict = get_hero_dict()

    for i in range(114):
        if i != 23:
            winrate_dict[hero_dict[i + 1]] = winrate[i]

    sorted_winrates = sorted(winrate_dict.items(), key=operator.itemgetter(1))
    x_plot_data = [x[0] for x in sorted_winrates]
    y_plot_data = [x[1] for x in sorted_winrates]

    title = 'Hero winrates at ' + mmr_info + ' MMR'
    data = [go.Bar(
        y=x_plot_data,
        x=y_plot_data,
        orientation='h'
    )]

    layout = go.Layout(
        title=title,
        width=1000,
        height=1400,
        yaxis=dict(title='hero',
                   ticks='',
                   nticks=114,
                   tickfont=dict(
                       size=8,
                       color='black')
                   ),
        xaxis=dict(title='win rate',
                   nticks=30,
                   tickfont=dict(
                       size=10,
                       color='black')
                   )
    )

    fig = go.Figure(data=data, layout=layout)

    py.iplot(fig, filename='hero_winrates_' + mmr_info)


def pick_statistics(dataset_df, mmr_info):
    x_data, y_data = dataset_df

    wins = np.zeros(114)
    games = np.zeros(114)
    pick_rate = np.zeros(114)

    for idx, game in enumerate(x_data):
        for i in range(228):
            if game[i] == 1:
                games[i % 114] += 1

                if y_data[idx] == 1:
                    if i < 114:
                        wins[i] += 1
                else:
                    if i >= 114:
                        wins[i - 114] += 1

    pick_rate = games / np.sum(games)

    pick_rate_dict = dict()
    hero_dict = get_hero_dict()

    for i in range(114):
        if i != 23:
            pick_rate_dict[hero_dict[i + 1]] = pick_rate[i]

    sorted_pickrates = sorted(pick_rate_dict.items(), key=operator.itemgetter(1))
    x_plot_data = [x[0] for x in sorted_pickrates]
    y_plot_data = [x[1] for x in sorted_pickrates]

    title = 'Hero pick rates at ' + mmr_info + ' MMR'
    data = [go.Bar(
        y=x_plot_data,
        x=y_plot_data * 100,
        orientation='h'
    )]

    layout = go.Layout(
        title=title,
        width=1000,
        height=1400,
        yaxis=dict(title='hero',
                   ticks='',
                   nticks=114,
                   tickfont=dict(
                       size=8,
                       color='black')
                   ),
        xaxis=dict(title='pick rate',
                   nticks=30,
                   tickfont=dict(
                       size=10,
                       color='black')
                   )
    )

    fig = go.Figure(data=data, layout=layout)

    py.iplot(fig, filename='hero_pickrates_' + mmr_info)


def mmr_distribution(csv_file):
    dataset = pd.read_csv(csv_file)

    data = [go.Histogram(x=dataset[:30000]['avg_mmr'])]

    layout = go.Layout(
        title='MMR distribution (sample of 30k games)'
    )

    fig = go.Figure(data=data, layout=layout)

    py.iplot(fig, filename='MMR_distribution')
