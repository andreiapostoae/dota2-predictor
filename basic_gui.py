from Tkinter import *
import ttk
import json
import logging
import pickle
import os
from os import listdir
from sklearn.externals import joblib
from query import process_query_list, give_result

MAX_MMR = 9000
MMR_INIT = 10000

heroes_json_data = {}

def get_hero_id(localized_name):
	for hero in heroes_json_data:
		if hero["localized_name"] == localized_name:
			return hero["id"]

def get_full_hero_list():
	global heroes_json_data
	json_data = json.load(open(os.path.join('preprocessing', 'heroes.json'), "rt"))

	hero_list = []
	heroes_json_data = json_data["heroes"]
	for hero in heroes_json_data:
		hero_list.append(hero["localized_name"])

	return hero_list

suggest_button = None
predict_button = None
boxes = []
unique_heroes = []
mmr_box = None
predict_result_label = None
suggest_result_label = None
no_mmr = True

def check_boxes_completed():
	global no_mmr
	global boxes
	global suggest_button
	global predict_button
	global mmr_box
	global unique_heroes

	completed_valid = 0
	total_completed = 0
	unique_heroes = []

	for box in boxes:
		box_text = box.get()

		if box_text != "":
			total_completed += 1

		if box_text != "" and box_text not in unique_heroes:
			unique_heroes.append(box_text)

	completed_valid = len(unique_heroes)

	if total_completed - completed_valid != 0:
		predict_button.config(state="disabled")
		suggest_button.config(state="disabled")
		return
			
	if completed_valid == 9 and no_mmr == False:
		suggest_button.config(state="normal")
		predict_button.config(state="disabled")
	elif completed_valid == 10  and no_mmr == False:
		predict_button.config(state="normal")
		suggest_button.config(state="disabled")
	else:
		predict_button.config(state="disabled")
		suggest_button.config(state="disabled")

def ComboBoxSelected(event):
	check_boxes_completed()


def validate(action, index, value_if_allowed, prior_value, text, validation_type, trigger_type, widget_name):
	global no_mmr

	if text in '0123456789':
		try:
			if value_if_allowed == "":
				no_mmr = True
				check_boxes_completed()
				return True

			val = int(value_if_allowed)
			if val >= 0 and val < 10000:
				no_mmr = False
				check_boxes_completed()
				return True
		except ValueError:
			return False
	
	return False


def process_query():
	global mmr_box
	global boxes
	global predict_result_label
	global suggest_result_label

	mmr = int(mmr_box.get())

	file_list = [int(valid_file[:-4]) for valid_file in listdir('pretrained') \
					if 'dicts' not in valid_file and 'results' not in valid_file]

	file_list.sort()

	min_distance = MMR_INIT
	final_mmr = MMR_INIT

	for model_mmr in file_list:
		if abs(mmr - model_mmr) < min_distance:
			min_distance = abs(mmr - model_mmr)
			final_mmr = model_mmr

	model = joblib.load("pretrained/" + str(final_mmr) + ".pkl")
	query_list = []

	faction = "Radiant"
	for i in range(10):
		name = boxes[i].get()
		if name != "":
			hero_id = get_hero_id(name)
			query_list.append(hero_id)
		else:
			if i < 5:
				faction = "Radiant"
			else:
				faction = "Dire"

	print query_list

	logging.basicConfig(level=logging.INFO, format='%(name)-10s %(levelname)-8s %(message)s')
	logger = logging.getLogger(__name__)

	if len(query_list) == 9:
		sorted_dict = process_query_list(query_list, heroes_json_data, faction, model, logger)
		i = 0

		label_text = ""
		for (hero, value) in sorted_dict:
			value = round(value, 2)

			hero_name = ""
			for json_hero in heroes_json_data:
				if json_hero["id"] == hero + 1:
					hero_name = json_hero["localized_name"]
					break

			label_text += "%s: %.2f%%\n" % (hero_name, value)
			i += 1
			if(i == 10):
				break 

		suggest_result_label['text'] = label_text
		predict_result_label['text'] = ""

	else:
		result = give_result(query_list, faction, model, logger)
		suggest_result_label['text'] = ""
		if result < 50.0:
			predict_result_label['text'] = "Dire has a %.2f%% chance to win" % (100 - result)
		else:
			predict_result_label['text'] = "Radiant has a %.2f%% chance to win" % result

def main():
	global suggest_button
	global predict_button
	global boxes
	global mmr_box
	global predict_result_label
	global suggest_result_label

	root = Tk()
	hero_list = get_full_hero_list()
	hero_list = sorted(hero_list)
	hero_list.insert(0, "")

	root.title("Dota 2 predictor")
	root.minsize(width=450, height=480)
	root.maxsize(width=450, height=480)

	radiant_label = Label(root, text="Radiant team")
	radiant_label.place(relx=0.15, rely=0.02)

	dire_label = Label(root, text="Dire team")
	dire_label.place(relx=0.67, rely=0.02)
	
	predict_button = Button(root, text="Predict winner", command=process_query)
	predict_button.place(relx=0.22, rely=0.45)
	predict_button.config(state="disabled")

	suggest_button = Button(root, text="Suggest hero", command=process_query)
	suggest_button.place(relx=0.5, rely=0.45)
	suggest_button.config(state="disabled")

	boxes = []

	for i in range(10):
		box_value = StringVar()
		box = ttk.Combobox(root, textvariable=box_value, state='readonly')

		box['values'] = hero_list
		box.current(0)

		box.place(relx=(i / 5) * 0.5 + 0.07, rely=(i % 5) * 0.05 + 0.1)
		box.bind("<<ComboboxSelected>>", ComboBoxSelected)
		boxes.append(box)

	avg_mmr = Label(root, text="Average MMR:")
	avg_mmr.place(relx=0.4, rely=0.4, anchor=CENTER)

	vcmd = (root.register(validate), '%d', '%i', '%P', '%s', '%S', '%v', '%V', '%W')
	mmr_box = Entry(root, validate="key", validatecommand=vcmd)
	mmr_box.place(relx=0.56, rely=0.4, anchor=CENTER, width=50)

	predict_result_label = Label(root, text="")
	predict_result_label.place(relx = 0.5, anchor=CENTER, y=350)

	suggest_result_label = Label(root, text="")
	suggest_result_label.place(relx = 0.5, rely=0.8, anchor=CENTER, )

	info_label1 = Label(root, text="For predicting the winner, select all the heroes in the game.")
	info_label1.place(x=10, y=250)

	info_label2 = Label(root, text="For getting last pick suggestions, select the other 9 heroes.")
	info_label2.place(x=10, y=270)

	label = Label(root, text="Andrei Apostoae, July 2017")
	label.place(x=300, y=460)
	label.configure(foreground="gray")

	label = Label(root, text="Current patch: 7.06d")
	label.place(x=10,y=460)
	label.configure(foreground="gray")

	root.mainloop()

if __name__ == "__main__":
	main()
