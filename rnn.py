import numpy as np

class RNNp:
    def __init__(self):
        """
        Insert correct values instead of the zeros below
        """
        self.Whh = np.array([[0,0,0],[0,0,0],[0,0,0]])
        self.Whx = np.array([[0,0,0],[0,0,0],[0,0,0]])
        self.Why = np.array([0,0,0])

    def activation(self, x):
        """
        Code an activation function appropriate for the problem
        """
        return x

    def add(self, a, b):
        h = np.array([0,0,0]).T
        A = [int(x) for x in np.binary_repr(a)]
        B = [int(x) for x in np.binary_repr(b)]
        A, B = self.harmonize(A,B)
        X = np.flipud(np.c_[A,B,np.ones(len(A),dtype=int)])
        r = ''
        h = np.array([0,0,0]).T
        for x in X:
            h = self.step(x,h)
            r += str(self.activation(self.Why@h))
        return int(r[::-1],2)
    
    def step(self, x, h):
        h = np.array([self.activation(x) for x in  self.Whh@h + self.Whx@x]).T
        return h

    def harmonize(self, a, b):
        maxlen = max(len(a), len(b))
        a = np.pad(a, (maxlen-len(a)+1,0), 'constant')
        b = np.pad(b, (maxlen-len(b)+1,0), 'constant')
        return a,b

"""
Testing may be realised as follows:

mrnn = RNNp()
for i in range(512):
    for j in range(256):
        r = mrnn.add(i,j)
        assert r == i+j, f"Addition is wrong as {i} + {j} != {r}"

"""