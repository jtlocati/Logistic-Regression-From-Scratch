from __future__ import annotations

import numpy

#Sigmoid Activation:
#Take in input array 'z' and return a % probibility
def sigmoid(z: numpy.ndarray) -> numpy.ndarray:
    z = numpy.asarray(z)
    out = numpy.empty_like(z, dtype=float)

    pos = z >= 0
    out[pos] = 1 / (1 + numpy.exp(-z[pos]))

    neg = ~pos
    ez = numpy.exp(z[neg])
    out[neg] = ez / (1+ez)

    return out

#Initalizes wights and bias
    # w = models weights
    # b = models bias
def initialize_parameters(n_features: int) -> tuple[numpy.ndarray, float]:

    w=numpy.zeros(n_features, dtype=float)
    b = 0
    return w,b

def forward(X: numpy.ndarray, w: numpy.ndarray, b: float) -> numpy.ndarray:

    z = X @ w+b #Calulate linear socre per-sample
    y_hat = sigmoid(z) #then use sigmoid to find probability

    return y_hat

#binary CE loss
def compute_loss(y_true: numpy.ndarray, y_pred: numpy.ndarray, eps: float = 1e-12) -> float:
    y_true = y_true.astype(float).reshape(-1)
    y_pred = numpy.clip(y_pred.astype(float).reshape(-1), eps, 1 - eps)


    losses = -(y_true * numpy.log(y_pred) + (1.0 - y_true) * numpy.log(1.0 - y_pred))

    return numpy.mean(losses)