import numpy as np

from lr_utils import load_dataset


def initialize(nx):
    w = np.zeros((nx, 1))
    b = np.zeros((1, 1))
    return { "w": w, "b": b }

def linear(x, w, b):
    return np.dot(w.T, x) + b

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def forward_prop(x, w, b):
    return sigmoid(linear(x, w, b))

def cross_entropy(a, y):
    m = y.shape[1]
    return -(1 / m) * np.sum(y * np.log(a) + (1 - y) * np.log(1 - a))

def backprop(x, y, a):
    m = x.shape[1]
    dw = (1 / m) * np.dot(x, (a - y).T)
    db = (1 / m) * np.sum(a - y)

    return {
        'dw': dw,
        'db': db,
    }

def update_params(lr, w, b, dw, db):
    w = w - lr * dw
    b = b - lr * db
    return {
        'w': w,
        'b': b,
    }

def optimize(x, y, lr, epoch):
    nx = x.shape[0]
    params = initialize(nx)

    for n in range(epoch):
        a = forward_prop(x, params["w"], params["b"])
        cost = cross_entropy(a, y)
        grads = backprop(x, y, a)
        params = update_params(lr, params["w"], params["b"], grads['dw'], grads['db'])

        if n % 100 == 0:
            print("Cost after iteration %i: %f" % (n, cost))

    return params

def predict(w, b, x):
    m = x.shape[1]
    prediction = np.zeros((1, m))
    w = w.reshape(x.shape[0], 1)

    a = sigmoid(np.dot(w.T, x) + b)

    for i in range(a.shape[1]):
        if a[0, i] > 0.5 :
            prediction[0,i] = 1
        else:
            prediction[0,i] = 0

    return prediction

def model(x_train, y_train, x_test, y_test, num_iterations=2000, learning_rate=0.5):
    params = optimize(x_train, y_train, learning_rate, num_iterations)

    w = params["w"]
    b = params["b"]

    y_prediction_test = predict(w, b, x_test)
    y_prediction_train = predict(w, b, x_train)

    print("Train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))
    print("Test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))

if __name__ == '__main__':
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

    train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
    test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

    train_set_x = train_set_x_flatten / 255.
    test_set_x = test_set_x_flatten / 255.

    model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2000, learning_rate=0.01)