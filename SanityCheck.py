import numpy
from model.logistic_regression import initialize_parameters, forward, compute_loss, TrainLoop, precision, accuracy, recall, predict
from results.Plotloss import save_loss_plot


def ds_forward_demo():
    X = numpy.array([
        [1.0, 0.0, 2.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 0.0],
        [2.0, 1.0, 1.0],
    ], dtype=float)
    y = numpy.array([1, 0, 1, 0], dtype=float)
    return X, y


def ds_separable():
    X = numpy.array([
        [1, 1],
        [2, 2],
        [-1, -1],
        [-2, -2]
    ], dtype=float)
    y = numpy.array([1, 1, 0, 0], dtype=float)
    return X, y


# -------------------------
# Minimal selection
# -------------------------

TestType = int(input("1-5: ").strip())


if TestType == 1:
    X, y = ds_forward_demo()
    w, b = initialize_parameters(X.shape[1])
    y_pred = forward(X, w, b)
    loss = compute_loss(y, y_pred)

    print("y_pred:", y_pred)
    print("loss:", loss)


elif TestType == 2:
    X, y = ds_separable()
    w, b, losses = TrainLoop(X, y, learning_rate=0.1, epochs=500)

    print("Final weights:", w)
    print("Final bias:", b)
    print("Final loss:", losses[-1])


elif TestType == 3:
    X, y = ds_separable()
    w, b, losses = TrainLoop(X, y, learning_rate=0.1, epochs=500)

    y_hat = predict(X, w, b)
    print("Accuracy:", accuracy(y, y_hat))
    print("Precision:", precision(y, y_hat))
    print("Recall:", recall(y, y_hat))


elif TestType == 4:
    X, y = ds_separable()
    w, b, losses = TrainLoop(X, y, learning_rate=0.1, epochs=500)

    save_loss_plot(losses)
    print("Saved loss plot.")


elif TestType == 5:
    X, y = ds_separable()

    w0, b0, losses0 = TrainLoop(X, y, learning_rate=0.1, epochs=2000, lambda_12=0.0)
    w1, b1, losses1 = TrainLoop(X, y, learning_rate=0.1, epochs=2000, lambda_12=0.5)

    print("||w|| no reg:", float(numpy.linalg.norm(w0)))
    print("||w|| L2 reg:", float(numpy.linalg.norm(w1)))