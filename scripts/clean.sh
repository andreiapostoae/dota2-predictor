#!/bin/bash

find .. -name "*.pyc" -type f -delete

if [ ! -f ../datasets.tar.gz ]; then
	tar -czvf ../datasets.tar.gz ../*.csv

rm -f ../*.csv

rm -f pylint.log

