from vanilla import AutoEncoder

# Don't hate me. You can just use other datasets.
from keras.datasets import mnist

import autograd.numpy as np
import matplotlib.pyplot as plt

(X_train, _), (X_test, _) = mnist.load_data()

X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.
X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
X_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))

enc = AutoEncoder()
enc.fit(X_train)

X_enc = enc.encode(X_test)._value
X_dec = enc.decode(X_enc)._value
plt.imshow(X_test[1000].reshape(28, 28))
plt.show()
plt.imshow(X_dec[1000].reshape(28, 28))