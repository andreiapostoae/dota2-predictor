import csv
import numpy as np

f = open("complete.csv", "rt")
csv_reader = csv.reader(f, delimiter=",")

output = open("complete_augmented.csv", "wt")
csv_writer = csv.writer(output, delimiter=',')

data_list = list(csv_reader)

for row in data_list:
	new_row = []
	new_row.append(int(row[0]))
	if row[1] == '0':
		new_row.extend([0, 1])
	else:
		new_row.extend([1, 0])

	new_row.extend(row[2:])

	#if int(row[13]) > 3000:
	csv_writer.writerow(new_row)

f.close()
output.close()
