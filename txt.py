import os
from glob import glob
import matplotlib.pyplot as plt
import scipy
pathes='C:\\Users\LENOVO\Desktop\data'
pathes=sorted(glob(pathes))
print(pathes)
#pathes=sorted(glob(os.path.join(file,'*.txt')))
#print(pathes)
for path in pathes:
    data_path = sorted(glob(os.path.join(path, '*.txt')))
    for f in data_path:
        #Data=[]
        with open(f,'r') as data:
            Data=data.readlines()
            Data=Data[0].split(" ")[0:-1]
            list=[]
            print(len(Data))
            for i in range(0,len(Data),2):
                if i<=len(Data)-2:
                    str=Data[i]+Data[i+1]
                    list.append(int(str,16))
            for i in range(len(list)):
                list[i]=list[i]-40960
                if list[i]<=0:
                    list[i]=0
                if list[i]>=4095:
                    list[i]=0
            print(list)
            plt.plot(list)
            print(f.replace(".txt",".jpg"))
            plt.savefig(f.replace(".txt",".jpg"))
            plt.clf()
