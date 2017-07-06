#!/usr/bin/env python
""" Prints linter details of a file in the project """

import os
from glob import glob
from pylint.lint import Run

def main():
	""" Main function """

	log_file = open("pylint.log", "wt")
	py_files = [y for x in os.walk("..") for y in glob(os.path.join(x[0], '*.py'))]

	for py_file in py_files:
		if "__init__.py" not in py_file:
			results = Run([py_file], exit=False)
			log_file.write(("%s: %.2f\n") % (py_file, results.linter.stats['global_note']))


	log_file.close()

if __name__ == "__main__":
	main()
