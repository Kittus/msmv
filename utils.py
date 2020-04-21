import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from scipy.stats import zscore

import models
from dirs import dir_features, dir_food101

"""
    INITIAL MODELS TRAINING FUNCTIONS
"""


def print_results_train(history, metrics, path='.', suffix=''):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    for metric in metrics:
        y = history.history[metric]
        x = range(len(y))

        plt.plot(x, y, 'b', label='Training {}'.format(metric))
        plt.title('Training {}'.format(metric))
        plt.legend()
        plt.savefig('{}/{}-{}.pdf'.format(path, metric, suffix))
        plt.close()


def print_results_train_val(history, metrics, path='.', suffix=''):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    for metric in metrics:
        y = history.history[metric]
        y2 = history.history['val_' + metric]
        x = range(len(y))

        plt.plot(x, y, 'b', label='Training {}'.format(metric))
        plt.plot(x, y2, 'r', label='Validation {}'.format(metric))
        plt.title('Training and validation {}'.format(metric))
        plt.legend()
        plt.savefig('{}/{}-{}.pdf'.format(path, metric, suffix))
        plt.close()


def save_model_and_weights(model, directory, filename='model'):
    model_json = model.to_json()
    with open(directory + filename + '.json', 'w') as json_file:
        json_file.write(model_json)
    model.save_weights(directory + filename + '.h5')


def get_data_generators(train_dir, test_dir, class_mode='categorical', batch_size=48, shuffle=True):
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       horizontal_flip=True,
                                       )

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        directory=train_dir,
        target_size=(224, 224),
        color_mode='rgb',
        shuffle=shuffle,
        batch_size=batch_size,
        class_mode=class_mode)

    test_generator = test_datagen.flow_from_directory(
        directory=test_dir,
        target_size=(224, 224),
        color_mode='rgb',
        shuffle=shuffle,
        batch_size=batch_size,
        class_mode=class_mode)

    return train_generator, test_generator


"""
    MULTI-SCALE MULTI-VIEW SYSTEM FUNCTIONS
"""


def fuse_features_scale(preds, fusion=np.max):
    return [fusion(preds[:, i]) for i in range(preds.shape[1])]


def generate_features(feature, scale, generate_labels=False):
    model_feat = models.load_feature_model(feature, scale)
    scale_modifier = pow(2, 2 * (scale - 1))
    train_dir = dir_food101 + 'flow{}/train/'.format(scale)
    test_dir = dir_food101 + 'flow{}/test/'.format(scale)
    batch_size = 101 * scale_modifier
    train_generator, test_generator = get_data_generators(train_dir, test_dir, batch_size=batch_size, shuffle=False)

    features = np.empty((0, model_feat.layers[-1].units))
    labels = np.empty((0, 101))

    for i in range(750):
        x, y = train_generator.next()
        preds = model_feat.predict(x)
        if scale > 1:
            preds = np.array([fuse_features_scale(pred) for pred in preds.reshape(-1, scale_modifier, preds.shape[1])])
        features = np.append(features, preds, axis=0)
        if generate_labels:
            labels = np.append(labels, y, axis=0)

    np.save(dir_features + feature + '_{}_train.npy'.format(scale), features)
    if generate_labels:
        np.save(dir_features + 'train_labels.npy', labels)

    features = np.empty((0, model_feat.layers[-1].units))
    labels = np.empty((0, 101))

    for i in range(250):
        x, y = test_generator.next()
        preds = model_feat.predict(x)
        if scale > 1:
            preds = np.array([fuse_features_scale(pred) for pred in preds.reshape(-1, scale_modifier, preds.shape[1])])
        features = np.append(features, preds, axis=0)
        if generate_labels:
            labels = np.append(labels, y, axis=0)

    np.save(dir_features + feature + '_{}_test.npy'.format(scale), features)
    if generate_labels:
        np.save(dir_features + 'test_labels.npy', labels)


def load_features(feature, scale, labels=True):
    x_train = np.load(dir_features + feature + '_{}_train.npy'.format(scale))
    x_test = np.load(dir_features + feature + '_{}_test.npy'.format(scale))
    if labels:
        y_train = np.load(dir_features + 'train_labels.npy')
        y_test = np.load(dir_features + 'test_labels.npy')
        return x_train, y_train, x_test, y_test
    else:
        return x_train, x_test


def load_feature_labels():
    y_train = np.load(dir_features + 'train_labels.npy')
    y_test = np.load(dir_features + 'test_labels.npy')
    return y_train, y_test


def get_ms_features(feature, scales=None, norm=zscore, labels=True):
    if scales is None:
        scales = [1, 2]
    features = [load_features(feature, scale, labels=False) for scale in scales]

    x_train = np.concatenate([norm(f[0], axis=1) for f in features], axis=1)
    x_test = np.concatenate([norm(f[1], axis=1) for f in features], axis=1)

    if labels:
        y_train, y_test = load_feature_labels()
        return x_train, y_train, x_test, y_test
    else:
        return x_train, x_test


def get_msmv_features(features, norm=zscore):
    features = [get_ms_features(feature, labels=False) for feature in features]

    x_train = np.concatenate([norm(f[0], axis=1) for f in features], axis=1)
    x_test = np.concatenate([norm(f[1], axis=1) for f in features], axis=1)

    y_train, y_test = load_feature_labels()
    return x_train, y_train, x_test, y_test
