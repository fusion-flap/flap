#!/bin/bash

# Error/warning codes to be ignored should be separated by '|' for 
# a valid regex expression.

IGNORE_LIST=":ES01:|:SA01:|:EX01:|:SS06:|:PR09:|:GL01:"

python -m numpydoc validate $1 | tail -n +4 | grep -v -E $IGNORE_LIST
