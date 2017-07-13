""" Repairs the mined data """
import pandas as pd

def main():
	""" main function """
	dataframe = pd.read_csv(sys.argv[1])

	cols = dataframe.columns.tolist()

	cols = [cols[0]] + [cols[-1]] + cols[1:-1]

	dataframe = dataframe[cols]

	dataframe.to_csv("repaired" + sys.argv[1], sep=',', index=False)

if __name__ == "__main__":
	main()

