#given a list of csv files, split them in half and save them on Ferreira/train/ and Ferreira/test/
# Usage: python split.py 
import os
local_path = os.getcwd()
files = ['ferreira9.csv','ferreira10.csv',
            'ferreira11.csv','ferreira12.csv',
            'ferreira13.csv','ferreira14.csv',
            'ferreira15.csv','ferreira16.csv']
