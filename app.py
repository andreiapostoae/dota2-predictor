from flask import Flask, render_template, request
import numpy as np, json, os
from training.query import query

app = Flask(__name__)

#main functionality
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == "POST":
        #do back end things here and then return what we want, take it back in a success function in JS and update the page.
        synergies = np.loadtxt('pretrained/synergies_all.csv')
        counters = np.loadtxt('pretrained/counters_all.csv')
        similarities = np.loadtxt('pretrained/similarities_all.csv')

        heroes = [name.encode('UTF8') for name in request.json['heroes']]

        print(heroes)
        radiant = [get_hero_id(hero) for hero in heroes[:5] if get_hero_id(hero)]
        dire = [get_hero_id(hero) for hero in heroes[5:] if get_hero_id(hero)]


        #text = query(request.json['mmr'], heroes)
        mmr = int(request.json['mmr'])
        #print(mmr, hero_ids[:5], hero_ids[5:])


        print(radiant, dire)
        text = query(mmr, radiant, dire, synergies, counters, similarities)
        #print text
        if isinstance(text, list):
            text = [str(hero[0]) + ': ' + str(round(hero[1][0] * 100, 2))+'% win rate. <br>' for hero in text[:10]]
            text = ''.join(text)

        print text
        return text

    hero_names = get_full_hero_list()
    radiant_heroes, dire_heroes = get_hero_factions()
    edited_names = [name.replace(" ", "_").replace("\'", "").lower() for name in hero_names]
    return render_template('main2.html', hero_names=sorted(hero_names), edited_names=sorted(edited_names), radiant_heroes=radiant_heroes, dire_heroes=dire_heroes)

def get_full_hero_list():

	hero_list = []
	heroes_json_data = json_data["heroes"]
	for hero in heroes_json_data:
		hero_list.append(hero["name"])

	return hero_list

#gets hero factions and primary attribute from new json file
def get_hero_factions():

    attributes = ["str", "agi", "int"]
    radiant_heroes, dire_heroes = {}, {}
    for attr in attributes:
        radiant_heroes[attr] = []
        dire_heroes[attr] = []

    for hero in json_data["heroes"]:
        if hero['faction'] == "Radiant":
            radiant_heroes[hero['primary_attribute'][:3].lower()].append(hero["name"].encode('UTF8'))
        else:
            dire_heroes[hero['primary_attribute'][:3].lower()].append(hero["name"].encode('UTF8'))


    return radiant_heroes, dire_heroes

def get_hero_id(name):
    for hero in json_data["heroes"]:
        if hero["name"] == name:
            return hero["id"]

if __name__ == '__main__':
    global json_data
    json_data = json.load(open(os.path.join('metadata.json'), "rt"))
    #app.run(debug= True,host="127.0.0.1",port=5000, threaded=True)
    app.run(debug=False,host="0.0.0.0",port=5000, threaded=True)
