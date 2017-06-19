import csv
import numpy as np
#import plotly.plotly as py
#import plotly.graph_objs as go
#import plotly
from matplotlib import pyplot as plt
from matplotlib import cm as cm
from matplotlib import mlab as ml

#plotly.tools.set_credentials_file(username='apostoae.andrei', api_key='CfhKA2mzZQZPc8nvnKjn')

f = open('706d.csv', 'rt')
csv_reader = csv.reader(f, delimiter=",")
full_list = list(csv_reader)

synergy_appearances_radiant = np.zeros((114, 114))
synergy_wins_radiant = np.zeros((114, 114))
synergy_winrate_radiant = np.zeros((114, 114))

synergy_appearances_dire = np.zeros((114, 114))
synergy_wins_dire = np.zeros((114, 114))
synergy_winrate_dire = np.zeros((114, 114))

counter_appearances = np.zeros((114, 114))
counter_wins = np.zeros((114, 114))
counter_winrate = np.zeros((114, 114))

for i in range(200000):
	game = full_list[i]
	for j in range(5):
		for k in range(5):
			if j != k:
				synergy_appearances_radiant[int(game[2+j]) - 1][int(game[2+k]) - 1] += 1
				synergy_appearances_dire[int(game[7+j]) - 1][int(game[7+k]) - 1] += 1

				if(int(game[1]) == 1):
					synergy_wins_radiant[int(game[2+j]) - 1][int(game[2+k]) - 1] += 1
					
				else:
					synergy_wins_dire[int(game[7+j]) - 1][int(game[7+k]) - 1] += 1
			
			counter_appearances[int(game[2+j]) - 1][int(game[7+k]) - 1] += 1
			if(int(game[1]) == 1):
				counter_wins[int(game[2+j]) - 1][int(game[7+k]) - 1] += 1				


for i in range(114):
	for j in range(114):
		if i != j:
			if synergy_appearances_radiant[i][j] == 0.0:
				synergy_winrate_radiant[i][j] = 0
			else:
				synergy_winrate_radiant[i][j] = synergy_wins_radiant[i][j] / synergy_appearances_radiant[i][j]

			if synergy_appearances_dire[i][j] == 0.0:
				synergy_winrate_dire[i][j] = 0
			else:
				synergy_winrate_dire[i][j] = synergy_wins_dire[i][j] / synergy_appearances_dire[i][j]

			if counter_appearances[i][j] == 0.0:
				counter_winrate[i][j] = 0
			else:
				counter_winrate[i][j] = counter_wins[i][j] / counter_appearances[i][j]
	
			
			

#print synergy_wins_radiant[63][90]
#print synergy_appearances_radiant[63][90]
#print counter_appearances[47][4]

#print counter_wins[47][34]
#print counter_appearances[47][34]

#trace = go.Heatmap(z=synergy_winrate_radiant)
#data = [trace]
#py.iplot(data, filename = 'heatmap')

H = counter_winrate

fig = plt.figure(figsize=(15, 15))

ax = fig.add_subplot(111)
ax.set_title('colorMap')
plt.imshow(H, interpolation = 'nearest', cmap = cm.coolwarm)
ax.set_aspect('equal')

cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
cax.get_xaxis().set_visible(False)
cax.get_yaxis().set_visible(False)
cax.patch.set_alpha(0)
cax.set_frame_on(False)
plt.colorbar(orientation='vertical')
plt.show()