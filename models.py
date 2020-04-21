import numpy as np
import tensorflow as tf
import keras

from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, BatchNormalization, ZeroPadding2D, \
    AveragePooling2D, GlobalAveragePooling2D, Input, Activation, Lambda, Layer
from keras import Model
from keras.optimizers import Adam, SGD
from keras.applications import VGG16
from keras.callbacks import LearningRateScheduler
from sklearn.preprocessing import MultiLabelBinarizer

import utils
import metrics
from dirs import dir_models, dir_food101, dir_ingredients, dir_results


def load_model(path):
    json_file = open(path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json, custom_objects={'softmax': tf.nn.softmax})
    return loaded_model


def load_feature_model(feature, scale):
    # Model to obtain features
    if feature == 'mlf':
        model_feat = load_model(dir_models + "model_ML_{}.json".format(scale))
        model_feat.load_weights(dir_models + "model_ML_{}.h5".format(scale))
    else:
        model_feat = load_model(dir_models + "model_SL_{}.json".format(scale))
        model_feat.load_weights(dir_models + "model_SL_{}.h5".format(scale))

    if feature == 'df':
        model_feat = Model(inputs=model_feat.input, outputs=model_feat.layers[-2].output)

    return model_feat


class StepDecay:
    def __init__(self, lr=0.0001, epoch_limit=10, factor=0.1):
        self.lr = lr
        self.epoch_limit = epoch_limit
        self.factor = factor

    def __call__(self, epoch):
        return self.lr if epoch <= self.epoch_limit else self.lr * self.factor


def add_weight_decay(model, weight_decay):
    alpha = weight_decay
    for layer in model.layers:
        if isinstance(layer, keras.layers.Conv2D) or isinstance(layer, keras.layers.Dense):
            layer.add_loss(keras.regularizers.l2(alpha)(layer.kernel))
        if hasattr(layer, 'bias_regularizer') and layer.use_bias:
            layer.add_loss(keras.regularizers.l2(alpha)(layer.bias))
    return model


def dish_model(n_classes):
    # vgg_conv = ResNet152(weights='imagenet', include_top=True, input_shape=(224, 224, 3))
    vgg_conv = VGG16(weights='imagenet', include_top=True, input_shape=(224, 224, 3))
    # vgg_conv = load_model("../.keras/models/vgg16_imagenet.json")
    # vgg_conv.load_weights("../.keras/models/vgg16_imagenet.h5")

    # # Freeze the layers
    # for layer in vgg_conv.layers[:-2]:
    #     layer.trainable = False

    x = vgg_conv.layers[-2].output
    predictions = Dense(n_classes, activation=tf.nn.softmax)(x)
    model = Model(inputs=vgg_conv.input, outputs=predictions)
    model = add_weight_decay(model, 0.0001)
    model.summary()

    # Compile the model
    sgd = SGD(0.0001, momentum=0.9)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['acc', keras.metrics.top_k_categorical_accuracy])
    return model


def ingredient_model(n_classes):
    vgg_conv = VGG16(weights='imagenet', include_top=True, input_shape=(224, 224, 3))
    # vgg_conv = load_model("../.keras/models/vgg16_imagenet.json")
    # vgg_conv.load_weights("../.keras/models/vgg16_imagenet.h5")

    # # Freeze the layers
    # for layer in vgg_conv.layers[:-2]:
    #     layer.trainable = False

    x = vgg_conv.layers[-2].output
    predictions = Dense(n_classes, activation='sigmoid')(x)
    model = Model(inputs=vgg_conv.input, outputs=predictions)
    model = add_weight_decay(model, 0.0001)
    model.summary()

    # Compile the model
    metrics_list = [metrics.f1]
    sgd = SGD(0.0001, momentum=0.9)
    model.compile(loss='binary_crossentropy',
                  optimizer=sgd,
                  metrics=metrics_list)
    return model


def fit_dish(scale, n_classes=101, epochs=30):
    train_dir = dir_food101 + 'flow{}/train/'.format(scale)
    test_dir = dir_food101 + 'flow{}/test/'.format(scale)

    train_generator, test_generator = utils.get_data_generators(train_dir, test_dir)
    model = dish_model(n_classes)

    steps = 75750 * pow(2, 2*(scale-1)) / train_generator.batch_size
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=steps,
        epochs=epochs,
        validation_data=test_generator,
        validation_steps=steps/3,
        callbacks=[LearningRateScheduler(StepDecay())],
        verbose=1)

    utils.print_results_train_val(history, ['acc', 'top_k_categorical_accuracy', 'loss'], dir_results)
    utils.save_model_and_weights(model, dir_models, filename='model_SL_{}'.format(scale))
    score, acc, top5acc = model.evaluate_generator(test_generator, steps=steps/3)
    print('Test score:{}\nTest accuracy:{}\nTest top-5 accuracy:{}\n'.format(score, acc, top5acc))

    return score, acc


def multilabel_flow_from_directory(flow_from_directory_gen, conversion, classes):
    for x, y in flow_from_directory_gen:
        yield x, np.array([conversion[classes[np.argmax(t)]] for t in y])


def fit_ingredient(scale, n_classes=227, epochs=30):
    train_dir = dir_food101 + 'flow{}/train/'.format(scale)
    test_dir = dir_food101 + 'flow{}/test/'.format(scale)

    annotations_dir = dir_ingredients + 'annotations/'

    classes = open(annotations_dir + 'classes.txt', 'r').read().split('\n')[:-1]
    ingredients = open(annotations_dir + 'ingredients_simplified.txt', 'r').read().split('\n')[:-1]
    ingredients = [i.split(',') for i in ingredients]

    mlb = MultiLabelBinarizer()
    mlb.fit(ingredients)
    n_classes = len(mlb.classes_)
    ingredients = mlb.transform(ingredients)

    conversion = dict(zip(classes, ingredients))
    train_generator, test_generator = utils.get_data_generators(train_dir, test_dir)
    model = ingredient_model(n_classes)

    steps = 75750 * pow(2, 2 * (scale - 1)) / train_generator.batch_size
    history = model.fit_generator(
        multilabel_flow_from_directory(train_generator, conversion, classes),
        steps_per_epoch=steps,
        epochs=epochs,
        callbacks=[LearningRateScheduler(StepDecay())],
        verbose=1)

    utils.print_results_train(history, ['f1', 'loss'], path=dir_results)
    utils.save_model_and_weights(model, dir_models, filename='model_ML_{}'.format(scale))
    score, acc = model.evaluate_generator(multilabel_flow_from_directory(train_generator, conversion, classes), steps=steps/3)
    print('Test score:', score)
    print('Test f1:', acc)

    return score, acc


def feature_flow_from_directory(flow_from_directory_gen):
    for x, y in flow_from_directory_gen:
        yield x, y


def softmax_model(input_size, lr=0.001):
    model = Sequential()
    model.add(Dense(101, input_shape=(input_size,), activation=tf.nn.softmax,
                    kernel_regularizer=keras.regularizers.l2(0.0001),
                    bias_regularizer=keras.regularizers.l2(0.0001)))

    sgd = SGD(lr, momentum=0.9)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=[keras.metrics.categorical_accuracy, keras.metrics.top_k_categorical_accuracy])
    return model
