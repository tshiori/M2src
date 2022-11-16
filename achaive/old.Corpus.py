import numpy as np
import matplotlib.pyplot as plt
import glob
from gensim.models import word2vec
import sys
import csv
import os
import shutil 

database = open("../data/Miyazawa_database.csv","r")

#database.csvの読み込み
csv_obj = csv.reader(database,delimiter=",")
#小説以外のidリストの作成
id_list = []
for line in csv_obj:
#    print(line)
    if(line[3]=="other"):
       id = line[2]
       id_list.append(id)
#print(id_list)

cp_list = []
for id in id_list:
    file_name = glob.glob("../../MExperiment/lcorpus/"+id+"_*uNeologd")
    if(len(file_name)==1):
        if(os.path.exists(file_name[0])):
            cp_list.append(file_name[0])
        else:
            print(id+"'s file not found")
    else:
        print(id,file_name)

for file_name in cp_list:
    try:
        shutil.copy(file_name,"../data/other/")
    except:
        print("[Error] Copy"+file_name)

