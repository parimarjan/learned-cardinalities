from scipy.optimize import minimize
import numpy as np

# def objective_function(beta, X, Y):
    # error = loss_function(np.matmul(X,beta), Y)
    # return(error)

class CustomLinearModel:
    """
    Linear model: Y = XB, fit by minimizing the provided loss_function
    with L2 regularization
    """
    def __init__(self, loss_function,
                 X=None, Y=None, sample_weights=None, beta_init=None,
                 regularization=0.00012):
        self.regularization = regularization
        self.beta = None
        self.loss_function = loss_function
        self.beta_init = beta_init

        self.X = X
        self.Y = Y


    def predict(self, X):
        prediction = np.matmul(X, self.beta)
        return(prediction)

    def model_error(self):
        error = self.loss_function(
            self.predict(self.X), self.Y)
        return(error)

    def l2_regularized_loss(self, beta):
        self.beta = beta
        return(self.model_error())
               # sum(self.regularization*np.array(self.beta)**2))

    def fit(self, maxiter=2500):
        # Initialize beta estimates (you may need to normalize
        # your data and choose smarter initialization values
        # depending on the shape of your loss function)
        if type(self.beta_init)==type(None):
            # set beta_init = 1 for every feature
            self.beta_init = np.array([1]*self.X.shape[1])
        else:
            # Use provided initial values
            pass

        if self.beta!=None and all(self.beta_init == self.beta):
            print("Model already fit once; continuing fit with more itrations.")

        res = minimize(self.l2_regularized_loss, self.beta_init,
                       method='BFGS', options={'maxiter': 500})
        self.beta = res.x
        self.beta_init = self.beta
