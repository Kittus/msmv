import cv2
import numpy as np
import shutil
import os

from keras.utils.np_utils import to_categorical

from dirs import dir_food101


# Divides an image into 4 images
def ImageDivision4(img):
    img = np.array(img)
    rows, cols = img.shape[0], img.shape[1]
    i = img[:rows // 2, :cols // 2, :]
    img1 = cv2.resize(img[:rows // 2, :cols // 2, :], (cols, rows), interpolation=cv2.INTER_AREA)
    img2 = cv2.resize(img[:rows // 2, cols // 2:, :], (cols, rows), interpolation=cv2.INTER_AREA)
    img3 = cv2.resize(img[rows // 2:, :cols // 2, :], (cols, rows), interpolation=cv2.INTER_AREA)
    img4 = cv2.resize(img[rows // 2:, cols // 2:, :], (cols, rows), interpolation=cv2.INTER_AREA)

    return [img1, img2, img3, img4]


# Food 101 data preparation

def load_food101():
    # Directory definitions
    dir_images = dir_food101 + 'images/'
    dir_meta = dir_food101 + 'meta/'

    # Variable definitions
    classes = open(dir_meta + 'classes.txt', 'r').read().split('\n')[:-1]
    n_classes = len(classes)
    dishes_to_id = dict(zip(classes, range(n_classes)))

    train_dishes = open(dir_meta + 'train.txt', 'r').read().split('\n')[:-1]
    test_dishes = open(dir_meta + 'test.txt', 'r').read().split('\n')[:-1]

    # Arrays to store images with different scales
    train_images = np.array([cv2.imread(dir_images + dir_i + '.jpg') for dir_i in train_dishes])
    test_images = np.array([cv2.imread(dir_images + dir_i + '.jpg') for dir_i in test_dishes])

    print('X_train:', train_images.shape)
    print('X_test:', test_images.shape)

    train_images2 = np.array([ImageDivision4(img) for img in train_images]).reshape(-1, train_images.shape[1],
                                                                                    train_images.shape[2],
                                                                                    train_images.shape[3])
    test_images2 = np.array([ImageDivision4(img) for img in test_images]).reshape(-1, test_images.shape[1],
                                                                                    test_images.shape[2],
                                                                                    test_images.shape[3])
    print('X_train2:', train_images2.shape)
    print('X_test2:', test_images2.shape)

    # Arrays to store labels for different scales and single and multi-label cases
    train_labels = np.array([dish.split('/')[0] for dish in train_dishes])
    test_labels = np.array([dish.split('/')[0] for dish in test_dishes])
    train_labels2 = np.array([[label] * 4 for label in train_labels]).flatten()
    test_labels2 = np.array([[label] * 4 for label in test_labels]).flatten()

    # Transform labels to format readable by keras (categorical)
    train_labels_c = to_categorical([dishes_to_id[label] for label in train_labels], n_classes)
    test_labels_c = to_categorical([dishes_to_id[label] for label in test_labels], n_classes)

    train_labels2_c = to_categorical([dishes_to_id[label] for label in train_labels2], n_classes)
    test_labels2_c = to_categorical([dishes_to_id[label] for label in test_labels2], n_classes)

    print('y_train:', train_labels_c.shape)
    print('y_test:', test_labels_c.shape)

    print('y_train2:', train_labels2_c.shape)
    print('y_test2:', test_labels2_c.shape)


# Creates all necessary directories to use flow_from_directories on scales 1 and 2
def generate_food101_flow():
    # Directory definitions
    dir_images = dir_food101 + 'images/'
    dir_meta = dir_food101 + 'meta/'
    dir_flow1 = dir_food101 + 'flow1/'
    dir_flow2 = dir_food101 + 'flow2/'

    # Variable definitions
    classes = open(dir_meta + 'classes.txt', 'r').read().split('\n')[:-1]
    n_classes = len(classes)

    train_dishes = open(dir_meta + 'train.txt', 'r').read().split('\n')[:-1]
    test_dishes = open(dir_meta + 'test.txt', 'r').read().split('\n')[:-1]

    # Create classes folders in flow
    for c in classes:
        dir_flow1_class_train = dir_flow1 + 'train/' + c
        dir_flow1_class_test = dir_flow1 + 'test/' + c
        dir_flow2_class_train = dir_flow2 + 'train/' + c
        dir_flow2_class_test = dir_flow2 + 'test/' + c

        if not os.path.isdir(dir_flow1_class_train):
            os.mkdir(dir_flow1_class_train)
        if not os.path.isdir(dir_flow1_class_test):
            os.mkdir(dir_flow1_class_test)
        if not os.path.isdir(dir_flow2_class_train):
            os.mkdir(dir_flow2_class_train)
        if not os.path.isdir(dir_flow2_class_test):
            os.mkdir(dir_flow2_class_test)

    # Create flow1 and flow2 folders
    # train
    for dir_i in train_dishes:
        shutil.copy(dir_images + dir_i + '.jpg', dir_flow1 + 'train/' + dir_i + '.jpg')
        img = cv2.imread(dir_images + dir_i + '.jpg')
        imgs = ImageDivision4(img)
        for i in range(len(imgs)):
            cv2.imwrite(dir_flow2 + 'train/' + dir_i + '_' + str(i) + '.jpg', imgs[i])
    # test
    for dir_i in test_dishes:
        shutil.copy(dir_images + dir_i + '.jpg', dir_flow1 + 'test/' + dir_i + '.jpg')
        img = cv2.imread(dir_images + dir_i + '.jpg')
        imgs = ImageDivision4(img)
        for i in range(len(imgs)):
            cv2.imwrite(dir_flow2 + 'test/' + dir_i + '_' + str(i) + '.jpg', imgs[i])
