import numpy as np
from sklearn.datasets import load_digits
from scipy.spatial.distance import pdist
from sklearn.manifold.t_sne import _joint_probabilities
from scipy import linalg
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import squareform
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import seaborn as sns
import os


def save_plot(save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    fig = plt.gcf()

    fig.savefig(save_path, dpi=300,bbox_inches='tight')
    print('png saved in: ', save_path)

#sns.set(rc={'figure.figsize':(11.7,8.27)})
#sns.set(rc={'figure.figsize':(10,10)})
palette = sns.color_palette("bright",2)

modal1 = 'T+V+A Speaker Dependent'
modal2='T+V+A Speaker Independent'
modal3='heatmap Speaker Dependent'
modal4='heatmap Speaker Independent'
work_dir = '/m_fusion_data/'
path =  work_dir + f'representation/{modal1}.npz'
#print(path)
data = np.load('/home/srp/CSY/MUStARD-master/TSNE/SI/tsne+76.68539325842697.npz')
data0 = np.load('/home/srp/CSY/MUStARD-master/TSNE/SI/dist+76.96629213483146.npz')
data0=data0['dist']
#print(data.files)
class_name=['non-sarcastic','sarcastic']
X = data['repr']
y4 = data['label']
print(y4)
y4 = [class_name[yi] for yi in y4]
tsne = TSNE()
X_embedded = tsne.fit_transform(X)
# print(y4)
print(X_embedded.shape)
#ax=plt.axes()
#sns.set_theme(style="darkgrid")
#sns.scatterplot(X_embedded[:,0], X_embedded[:,1],hue=y4, legend='full', palette="Set2",s=200,alpha=.5)
#sns.histplot(x=X_embedded[:,0], y=X_embedded[:,1], bins=50, pthresh=.1, cmap="mako")
#sns.kdeplot(x=X_embedded[:,0], y=X_embedded[:,1], levels=5, color="w", linewidths=1)
#sns.heatmap(data0,cmap="YlGnBu")
sns.jointplot(X_embedded[:,0], X_embedded[:,1],hue=y4,palette="Set2",s=200,alpha=.5)
#ax.set_title(modal2)
plt.legend()
# plt.show()
#path =  work_dir + f'representation/img/{modal}.png'
save_plot(f'/home/srp/CSY/MUStARD-master/TSNE/image/SI+{modal2}.png')
