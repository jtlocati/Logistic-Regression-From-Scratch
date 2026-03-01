import numpy
from model.logistic_regression import initialize_parameters, forward, compute_loss

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

#inital logistic regression test as of 'addition of base logistic regression architecture.' comit:
    #y_pred: [0.5 0.5 0.5 0.5]
    #loss: 0.6931471805599453