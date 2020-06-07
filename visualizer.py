import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
import seaborn as sns
  
def visualize_embeddings(data_loader,gnet,device):
  
  overall_embeddings = None
  overall_labels = None
  for i,batch in enumerate(data_loader):
      X, y = batch[0].to(device), batch[1].to(device)
      embeddings, _ = gnet(X)
      embeddings = embeddings.detach().cpu().numpy()
      labels = y.detach().cpu().numpy()
      if overall_embeddings is None:
        overall_embeddings = embeddings
        overall_labels = list(labels)
      else:
        overall_embeddings = np.concatenate([overall_embeddings,embeddings],axis=0)
        overall_labels.extend(list(labels))
      if i > 5:
        break
  overall_labels = np.array(overall_labels)

  X = overall_embeddings
  X_embedded = TSNE(n_components=2).fit_transform(X)
  X_embedded.shape

  # plt.figure()

  # for l in range(10):
  #   plt.scatter(X_embedded[overall_labels==l,0],X_embedded[overall_labels==l,1])

  # plt.show()
  sns.set(rc={'figure.figsize':(11.7,8.27)})
  palette = sns.color_palette("bright", 10)
  plt.figure()
  sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=overall_labels, legend='full', palette=palette)
  plt.show()


def visualize_input(X):
  image = X.transpose(0,2).transpose(0,1).cpu().numpy()
  image = (image - np.min(image)) / (np.max(image) - np.min(image))

  plt.figure()
  plt.imshow(image)
  plt.show()