""" Repairs the mined data """
import pandas as pd

def main():
	""" main function """
	dataframe = pd.read_csv("part3.csv")

	cols = dataframe.columns.tolist()

	cols = [cols[0]] + [cols[-1]] + cols[1:-1]

	dataframe = dataframe[cols]

	dataframe.to_csv("part3_repaired.csv", sep=',', index=False)

if __name__ == "__main__":
	main()

