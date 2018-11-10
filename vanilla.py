from autograd import grad
from autograd.misc.optimizers import adam
import autograd.numpy as np

'''
Nonlinear activation functions
'''
def relu(X):
    return X * (X > 0)

def sig(X):
    return 1/(1+np.exp(-X))

'''
Useful functions for a smoother training
'''
def batch_normalize(W):
    mu = np.mean(W, axis=0)
    var = np.var(W, axis=0)
    W = (W - mu)/np.sqrt(var+1)
    return W

'''
Loss functions
'''
def rmse(X, X_prime):
    return np.mean((X-X_prime).sum(axis=0) ** 2)


''' 
Single-layered autoencoder
'''
class AutoEncoder:
    def __init__(self, learning_rate = 0.01, batch_size = 64, code_size = 32):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.code_size = code_size
    
    def fit(self, X):
        # Encoder weight & bias
        self.W = np.random.randn(X.shape[1], self.code_size)
        self.b = np.full(self.code_size, 0.1)
        
        # Decoder weight & bias
        self.W_prime = np.random.randn(self.code_size, X.shape[1])
        self.b_prime = np.full(X.shape[1], 0.1)
        
        # Group model parameters for later optimizations
        params = [self.W, self.b, self.W_prime, self.b_prime]
        
        # Make batches out of datasets
        batches = np.array_split(X, X.shape[0] // self.batch_size)
        
        # set the objective function
        def objective(params, step):
            self.W, self.b, self.W_prime, self.b_prime = params
            chunk = batches[int(step % len(batches))]
            C = self.encode(chunk)
            X_prime = self.decode(C)
            return rmse(chunk, X_prime)
        
        # Compute gradient of model parameters. Yes, we are not doing manual 
        # partial differentiation. No one sane does.
        objective_grad = grad(objective) # See? Science.
        max_epoch = 500
        
        def callback(params, step, g):
            if step % max_epoch == 0:
                print("Iteration {0:3d} objective {1:1.2e}".
                      format(step//max_epoch + 1, objective(params, step)))
        
        # The real optimization goes here
        params = adam(objective_grad, params, step_size = 0.01,
                      num_iters = 50 * max_epoch, callback = callback)
    
    def encode(self, X):
        W_norm = batch_normalize(self.W)
        nonlin = sig((X @ W_norm) + self.b.T)
        return nonlin
    
    def decode(self, X):
        nonlin = sig((X @ self.W_prime) + self.b_prime.T)
        return nonlin