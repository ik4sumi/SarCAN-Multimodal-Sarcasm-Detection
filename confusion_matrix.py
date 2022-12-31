import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
def plot(matrix):
  sns.set()
  f,ax=plt.subplots()
  print(matrix) #打印出来看看
  sns.heatmap(matrix,annot=True,cmap="Blues",fmt="d",ax=ax) #画热力图
  ax.set_title('Speaker Independent confusion matrix') #标题
  ax.set_xlabel('True') #x轴
  ax.set_ylabel('Predict') #y轴
  plt.savefig('/home/srp/CSY/MUStARD-master/TSNE/image/Speaker Independent confusion matrix-new.png')

matrix=np.array([[176,47],[28,105]])
plot(matrix)# 画原始的数据
plt.show()

