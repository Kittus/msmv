import utils
import models


# feature in {df, hlf, mlf}
# scales is array of size >= 1 of values {1, 2}
def multi_scale_feature_test(feature, scales, epochs=30, batch_size=32, lr=0.001):
    # Data loading
    x_train, y_train, x_test, y_test = utils.get_ms_features(feature, scales)

    # Model architecture and fit
    model = models.softmax_model(x_train.shape[1], lr=lr)
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=1,
                        validation_data=(x_test, y_test))

    # Additional result obtention
    utils.print_results_train_val(history, ['categorical_accuracy', 'top_k_categorical_accuracy', 'loss'],
                                  suffix='{}{}'.format(feature, scales))
    score, acc, top5acc = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)
    print('Test score:{}\nTest accuracy:{}\nTest top-5 accuracy:{}\n'.format(score, acc, top5acc))


# feature is array of size >= 1 of values {df, hlf, mlf}
def multi_view_feature_test(features, epochs=30, batch_size=32, lr=0.001):
    # Data loading
    x_train, y_train, x_test, y_test = utils.get_msmv_features(features)

    # Model architecture and fit
    model = models.softmax_model(x_train.shape[1], lr=lr)
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=1,
                        validation_data=(x_test, y_test))

    # Additional result obtention
    utils.print_results_train(history, ['categorical_accuracy', 'top_k_categorical_accuracy', 'loss'],
                              suffix='{}'.format(features))
    score, acc, top5acc = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)
    print('Test score:{}\nTest accuracy:{}\nTest top-5 accuracy:{}\n'.format(score, acc, top5acc))
