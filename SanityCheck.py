import numpy
from model.logistic_regression import initialize_parameters, forward, compute_loss, TrainLoop

TestType = int(input("test type"))


if TestType == 1:
    # Test DS: 4 samples, 3 features
    X = numpy.array([
        [1.0, 0.0, 2.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 0.0],
        [2.0, 1.0, 1.0],
    ])

    y = numpy.array([1, 0, 1, 0])

    w, b = initialize_parameters(n_features=X.shape[1])
    y_pred = forward(X, w, b)
    loss = compute_loss(y, y_pred)

    print("y_pred:", y_pred)
    print("loss:", loss)

    #inital logistic regression test as of 'addition of base logistic regression architecture.' commit:
        #y_pred: [0.5 0.5 0.5 0.5]
        #loss: 0.6931471805599453

elif TestType == 2:
    # Create simple linearly separable dataset
    X = numpy.array([
        [1, 1],
        [2, 2],
        [-1, -1],
        [-2, -2]
    ], dtype=float)

    y = numpy.array([1, 1, 0, 0])

    w, b, losses = TrainLoop(X, y, learning_rate=0.1, epochs=500)

    print("Final weights:", w)
    print("Final bias:", b)
    print("Final loss:", losses[-1])
    # Inital Training loop test @ 'Addition of gradient decent function and training loop for model' commit:
        #Final weights: [2.00009461 2.00009461]
        #Final bias: -1.0732517305434009e-16
        #Final loss: 0.009258349193133278