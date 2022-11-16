import Modules as m
import shutil
import glob
import os

#著者名リスト
authors = ['Akutagawa','Arisima','Kajii','Kikuchi','Sakaguchi','Dazai','Nakajima','Natsume','Makino','Miyazawa']
#authors = ['Akutagawa']

file_name="../debug/copyFromAozoraBunkodir_error.list"
m.resetFile(file_name)
f=open(file_name,"a")

for author in authors:
    index = 0
    auth_works,index_fin = m.getWorkObj(author,index)

    print("---------------"+author+"---------------",file=f)

    

    for i in range(0,len(auth_works)):
        file_path = glob.glob("../../AozoraBunko/"+auth_works[i].author+"/"+str(auth_works[i].id)+"_"+"*.txt")
        if(len(file_path)==0):
            print(str(auth_works[i].id)+"file not found",file=f)
            print("../../AozoraBunko/"+auth_works[i].author+"/"+str(auth_works[i].id)+"_"+"*.txt")
            exit()
        else:
            if(auth_works[i].novel==True):
                try:
                    shutil.copy(file_path[0],"../data/"+auth_works[i].author+"/novel2/")
                except:
                    os.mkdir("../data/"+auth_works[i].author+"/novel2/")
                    shutil.copy(file_path[0],"../data/"+auth_works[i].author+"/novel2/")
            elif(auth_works[i].novel==False):
                try:
                    shutil.copy(file_path[0],"../data/"+author+"/other2/")
                except:
                    os.mkdir("../data/"+auth_works[i].author+"/other2/")
                    shutil.copy(file_path[0],"../data/"+auth_works[i].author+"/other2/")    
            else:
                print(str(auth_works[i].id)+"object.novel is empty",file=f)

    
        