""" Module responsible with plotting of hero synergies and counters """
import numpy as np

import plotly.graph_objs as go
import plotly.plotly as py

from tools.metadata import get_hero_dict


def plot_synergies():
    synergies = np.loadtxt('pretrained/synergies_all.csv')

    for i in range(114):
        synergies[i, i] = 0.5

    hero_dict = get_hero_dict()

    x_labels = []
    for i in range(114):
        if i != 23:
            x_labels.append(hero_dict[i + 1])

    synergies = np.delete(synergies, [23], 0)
    synergies = np.delete(synergies, [23], 1)

    trace = go.Heatmap(z=synergies,
                       x=x_labels,
                       y=x_labels,
                       colorscale='Viridis')

    layout = go.Layout(
        title='Hero synergies',
        width=1000,
        height=1000,
        xaxis=dict(ticks='',
                   nticks=114,
                   tickfont=dict(
                        size=8,
                        color='black')),
        yaxis=dict(ticks='',
                   nticks=114,
                   tickfont=dict(
                        size=8,
                        color='black'))

    )

    data = [trace]
    fig = go.Figure(data=data, layout=layout)

    py.iplot(fig, filename='heatmap_synergies')


def plot_counters():
    counters = np.loadtxt('pretrained/counters_all.csv')

    for i in range(114):
        counters[i, i] = 0.5

    hero_dict = get_hero_dict()

    x_labels = []
    for i in range(114):
        if i != 23:
            x_labels.append(hero_dict[i + 1])

    counters = np.delete(counters, [23], 0)
    counters = np.delete(counters, [23], 1)

    trace = go.Heatmap(z=counters,
                       x=x_labels,
                       y=x_labels,
                       colorscale='Viridis')

    layout = go.Layout(
        title='Hero counters (hero1 winrate against hero2)',
        width=1000,
        height=1000,
        xaxis=dict(ticks='',
                   nticks=114,
                   title='hero2',
                   tickfont=dict(
                        size=8,
                        color='black')),
        yaxis=dict(ticks='',
                   nticks=114,
                   title='hero1',
                   tickfont=dict(
                        size=8,
                        color='black'))

    )

    data = [trace]
    fig = go.Figure(data=data, layout=layout)

    py.iplot(fig, filename='heatmap_counters')
