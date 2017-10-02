import numpy as np
import numpy.random as rng
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from sklearn.metrics import pairwise_distances_argmin

n_colors=32
pic='sutd.png'
img=mpimg.imread(pic)
img=img[:,:,:3]


w,h,d=tuple(img.shape)
image_array=np.reshape(img,(w*h,d))
image_array_sample = shuffle(image_array, random_state=0)[:1000]

#Kmeans and its labels
kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)
labels = kmeans.predict(image_array)

#Random and its labels
codebook_random = shuffle(image_array, random_state=0)[:n_colors + 1]
labels_random = pairwise_distances_argmin(codebook_random,image_array,axis=0)

def recreate_image(palette,labels,w,h):
    d=palette.shape[1]
    image=np.zeros((w,h,d))
    label_idx=0
    for i in range(w):
        for j in range(h):
            image[i][j]=palette[labels[label_idx]]
            label_idx+=1
    return image

plt.figure(1)
plt.clf()
ax=plt.axes([0,0,1,1])
plt.axis('off')
plt.title('Originalimage(16.8millioncolors)')
plt.imshow(img)

plt.figure(2)
plt.clf()
ax=plt.axes([0,0,1,1])
plt.axis('off')
plt.title('Compressedimage(K-Means)')
plt.imshow(recreate_image(kmeans.cluster_centers_,labels,w,h))

plt.figure(3)
plt.clf()
ax=plt.axes([0,0,1,1])
plt.axis('off')
plt.title('Compressedimage(Random)')
plt.imshow(recreate_image(codebook_random,labels_random,w,h))
plt.show()