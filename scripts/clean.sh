#!/bin/bash

find .. -name "*.pyc" -type f -delete
find .. -name "complete.csv" -type f -delete
rm -f pylint.log

