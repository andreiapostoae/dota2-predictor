""" Converts the y column in one hot format for the NN notebook """
import csv
import sys

def main():
	""" Main function """
	input_file = open(sys.argv[1], "rt")
	csv_reader = csv.reader(input_file, delimiter=",")

	output = open("augmented_" + sys.argv[1], "wt")
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

		csv_writer.writerow(new_row)

	input_file.close()
	output.close()

if __name__ == "__main__":
	main()
