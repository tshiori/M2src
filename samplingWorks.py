import Modules as m
import random
import shutil
import glob

#authors=["Akutagawa","Sakaguchi","Miyazawa"]
authors=["Akutagawa","Sakaguchi"]
#authors=["Miyazawa"]

cate1=913
cate2=914
set_name="set1"

for author in authors:
    index = 0
    auth_works,index_fin = m.getWorkObj(author,index)

    #print(len(auth_works))
    '''
    for i in range(0,10):
        auth_works[i].PrintSelfInformation()
    '''

    parent_set=[]
    for i in range(0,len(auth_works)):
        if(auth_works[i].duplicate==False):
            if(auth_works[i].category==cate1 or auth_works[i].category==cate2):
                parent_set.append(i)

    #print(len(parent_set))

    for i in parent_set:
        if(auth_works[i].category==cate1 or auth_works[i].category==cate2):
            pass
        else:
            print("ERROR")
            auth_works[i].PrintSelfInformation

    samples = random.sample(parent_set,10)

    file_name="../human_data/works/"+author+"/"+set_name+"/"+set_name+".list"
    m.resetFile(file_name)
    f=open(file_name,"a")
    for i in samples:
        auth_works[i].PrintSelfInformation_file(f)
        print("",file=f)
        #print("../../MExperiment/data/"+auth_works[i].author+"/"+str(auth_works[i].id)+"_"+"*.txt")
        file_path = glob.glob("../../MExperiment/data/"+auth_works[i].author+"/"+str(auth_works[i].id)+"_"+"*.txt")
        if(len(file_path)==0):
           file_path = glob.glob("../../MExperiment/lcorpus/"+str(auth_works[i].id)+"_"+"*.txt")
        if(len(file_path)==0):
            print(str(auth_works[i].id))
        else:
            shutil.copy(file_path[0],"../human_data/works/"+author+"/"+set_name+"/")
        