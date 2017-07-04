#!/bin/bash

find .. -name "*.pyc" -type f -delete
tar -czvf ../datasets.tar.gz ../*.csv
rm ../*.csv
rm -f pylint.log

