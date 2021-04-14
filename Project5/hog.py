from skimage.feature import hog
from sklearn.datasets import fetch_lfw_people
from sklearn import svm
from sklearn.model_selection import train_test_split

def get_result(kernel_:str, x_train, x_test, y_train, y_test, c:int, gamma_:str):
    clf = svm.SVC(kernel = kernel_, C = c, gamma = gamma_)
    clf.fit(x_train, y_train)

    print(clf.predict(x_test))
    print("Accuracy: ", clf.score(x_test, y_test))

lfw_people = fetch_lfw_people(min_faces_per_person = 200, resize = 0.4)
image_people = lfw_people['images']
target_people = lfw_people['target']

hogged_people = []
for image in image_people:
    fd, hog_image = hog(image, orientations = 8, pixels_per_cell = (9, 9),
                        cells_per_block=(1, 1), visualize=True, multichannel = False)
    hogged_people.append(fd)

# split some data to be the test data
x_train, x_test, y_train, y_test = train_test_split(hogged_people, target_people, test_size = 0.2, random_state = 0)

get_result('linear', x_train, x_test, y_train, y_test, 1, 'auto')