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
def compute_loss(y_true, y_pred, w=None, lambda_12: float = 0.0, eps=1e-12) -> float:
    y_true = y_true.astype(float).reshape(-1)
    y_pred = numpy.clip(y_pred.astype(float).reshape(-1), eps, 1 - eps)


    losses = -(y_true * numpy.log(y_pred) + (1.0 - y_true) * numpy.log(1.0 - y_pred))

    # Penalize large weights to protect over fitting by keeping weights small
    if w is not None and lambda_12 > 0.0:
        m = y_true.shape[0]
        penalty_12 = lambda_12 / (2*m) * numpy.sum(w**2)
        return (numpy.mean(losses) + penalty_12)

    return numpy.mean(losses)


#calculate the graidents of loss w.r.t to then update weights and bias
def gradient(X: numpy.ndarray, y_true: numpy.ndarray, y_pred: numpy.ndarray, w, lambda_12: float = 0) -> tuple[numpy.ndarray, float]:

    m = X.shape[0] #number of samples

    # calculate error per sample, prediction too high dz is positive & vice-versa
    dz = y_pred - y_true

    # weight and bias gradient values, detirmines by how much is a weight is edited 
    dw = (X.T @ dz) / m

    #L2 Gradiant
    if lambda_12 > 0:
        m = X.shape[0]
        dw = dw + (lambda_12 / m) * w
    db = numpy.sum(dz) / m

    return dw, db

#Train the model usign gradient decent
# w = learned weights, b = learned bias, losses = array of lost values over time 
def TrainLoop(X: numpy.ndarray, Y: numpy.ndarray, learning_rate=0.1, epochs=1000, lambda_12: float = 0.0) -> tuple[numpy.ndarray, float, list[float]]:


    #initalize params
    Y = Y.reshape(-1) #Ensue that y is 1D
    w, b = initialize_parameters(X.shape[1])
    losses = []

    step = max(1, epochs // 10)
    for epoch in range(epochs):
        y_pred = forward(X, w, b)

        #Claculate loss
        loss = compute_loss(Y, y_pred, w=w, lambda_12=lambda_12)
        losses.append(loss)

        #Backdrop
        dw, db = gradient(X, Y, y_pred, w=w, lambda_12=lambda_12)

        #update values by amount specifyed in gradient
        w = w - learning_rate * dw
        b = b - learning_rate * db

        if DEBUG:
            if epoch % step == 0:
                print(f"Epoch = {epoch} && Loss = {loss:.6f}")

    return w, b, losses

#Predict probabilitys of 'm' for class one (essencally repackageing the forward function, this just allows function to make more sence when used in code)
def probability(X: numpy.ndarray, w: numpy.ndarray, b: float) -> numpy.ndarray:
    return forward(X, w, b)

#convert probabilitys into boolean
def predict(X: numpy.ndarray, w: numpy.ndarray, b: float, threshold: float = 0.0) -> numpy.ndarray:
    probabilitys = probability(X, w, b)
    return (probabilitys >= threshold).astype(int) #converts into binary dependent on threshold sensitvity


#---------METRICS------------

def accuracy(y_true: numpy.ndarray, y_prediction: numpy.ndarray) -> float:
    y_true = y_true.reshape(-1)
    y_prediction = y_prediction.reshape(-1)
    return float(numpy.mean(y_true == y_prediction))

def precision(y_true: numpy.ndarray, y_prediction: numpy.array) -> float:
    y_true = y_true.reshape(-1)
    y_prediction = y_prediction.reshape(-1)

    #True & False positives (relate in: {Predicted value} and {Actual Value})
    tp = numpy.sum((y_true == 1) & (y_prediction == 1))
    fp = numpy.sum((y_true == 0) & (y_prediction == 1))
    return float(tp / (tp + fp + 1e-12))

def recall(y_true: numpy.ndarray, y_prediction: numpy.ndarray) -> float:
    y_true = y_true.reshape(-1)
    y_prediction = y_prediction.reshape(-1)

    tp = numpy.sum((y_true == 1) & (y_prediction == 1))
    fn = numpy.sum((y_true == 1) & (y_prediction == 0))
    return float(tp / (tp + fn + 1e-12))
