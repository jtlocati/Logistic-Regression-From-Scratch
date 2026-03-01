from __future__ import annotations

import numpy

#optional debug setting for prin step in TrainLoop
DEBUG = False

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


#calculate the graidents of loss w.r.t to then update weights and bias
def gradient(X: numpy.ndarray, y_true: numpy.ndarray, y_pred: numpy.ndarray) -> tuple[numpy.ndarray, float]:

    m = X.shape[0] #number of samples

    # calculate error per sample, prediction too high dz is positive & vice-versa
    dz = y_pred - y_true

    # weight and bias gradient values, detirmines by how much is a weight is edited 
    dw = (X.T @ dz) / m
    db = numpy.sum(dz) / m

    return dw, db

#Train the model usign gradient decent
# w = learned weights, b = learned bias, losses = array of lost values over time 
def TrainLoop(X:numpy.ndarray, Y:numpy.ndarray, learning_rate: float = 0.1, epochs: int = 1000) -> tuple[numpy.ndarray, float, list[float]]:


    #initalize params
    Y = Y.reshape(-1) #Ensue that y is 1D
    w, b = initialize_parameters(X.shape[1])
    losses = []

    step = max(1, epochs // 10)
    for epoch in range(epochs):
        y_pred = forward(X, w, b)

        #Claculate loss
        loss = compute_loss(Y, y_pred)
        losses.append(loss)

        #Backdrop
        dw, db = gradient(X, Y, y_pred)

        #update values by amount specifyed in gradient
        w = w - learning_rate * dw
        b = b - learning_rate * db

        if DEBUG:
            if epoch % step == 0:
                print(f"Epoch = {epoch} && Loss = {loss:.6f}")

    return w, b, losses