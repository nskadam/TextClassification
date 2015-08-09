# -*- coding: utf-8 -*-
"""
Created on Sat Aug 08 10:38:52 2015

@author: nilesh
"""

import os
import pandas as pd

# User Defined Veriables ------------------------------------------------------
path_base = 'E:/OneDrive/2. Projects/7. Perosonal/59. Text Classification 20 news'
dir_data = '1. Data'
dir_data_proc = '2. Data Processing'

def files_to_csv():
    out = []
    for fld in os.listdir(os.path.join(path_base, dir_data, '20news-bydate')):
        print fld
        for fl in os.listdir(os.path.join(path_base, dir_data, '20news-bydate', fld)):
            with open(os.path.join(path_base, dir_data, '20news-bydate', fld, fl), 'r') as fl_in:                
                try:                
                    out_part = fl_in.read().replace('\n', '').encode('utf-8')       
                except UnicodeDecodeError:
                    out_part = 'Error while encoding'                    
            out.append([fl, fld, out_part])
    out = pd.DataFrame(out, columns = ['id', 'category', 'text'])
    out.to_csv(os.path.join(path_base, dir_data_proc, 'a_a_20news.csv'), index = False)

os.chdir(path_base)
files_to_csv()





