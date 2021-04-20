import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from scipy.cluster.vq import *

def get_image_list(amount, type):
    image_list = []

    for x in range(1, amount + 1):
        if type == 'cat':
            img = cv2.imread('./datasets/training_set/cats/cat.%d.jpg' % x)
        elif type == 'dog':
            img = cv2.imread('./datasets/training_set/dogs/dog.%d.jpg' % x)
        
        image_list.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    
    return image_list

def get_result(x_train, x_test, y_train, y_test):
    clf = LinearSVC()
    clf.fit(x_train, y_train)

    return round(clf.score(x_test, y_test), 4)

# cat 0 dog 1
num_of_cats = 200
num_of_dogs = 200
data_size = num_of_cats + num_of_dogs

ml_data = get_image_list(num_of_cats, 'cat') + get_image_list(num_of_dogs, 'dog')
ml_target = [0] * num_of_cats + [1] * num_of_dogs

sift = cv2.SIFT_create()
des_list = []
for data in ml_data:
    data = cv2.resize(data, (300, 300))
    kpts = sift.detect(data)
    _, des = sift.compute(data, kpts)
    des_list.append(des)

descriptors = des_list[0]
for descriptor in des_list[1:]:
    descriptors = np.vstack((descriptors, descriptor))

k_means = 100
voc, variance = kmeans(descriptors, k_means, 1)
im_features = np.zeros((data_size, k_means), 'float32')
for i in range(data_size):
    words, distance = vq(des_list[i], voc)
    for word in words:
        im_features[i][word] += 1

print('data_size =', data_size)

# split some data to be the test data
result = train_test_split(im_features, ml_target, test_size = 0.2, random_state = 0)
print('Accuracy =', get_result(*result))