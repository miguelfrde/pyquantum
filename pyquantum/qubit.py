from functools import reduce
import numpy as np

class Qubit:
    KET = True
    BRA = False
    
    def __init__(self, n, state=KET):
        if state != Qubit.KET and state != Qubit.BRA:
            raise ValueError("State must be either KET or BRA")
        self.vector = np.matrix([[1], [0]] if n == 0 else [[0], [1]])
        self.state = state

    def __new(self, vector, state=KET):
        q = Qubit(1, state)
        q.vector = vector
        return q

    def conjugate(self):
        return self.__new(np.transpose(np.conjugate(self.vector)),
                not self.state)

    def tensor(self, other):
        return self.__new(np.kron(self.vector, other.vector), self.state)
   
    def apply_gate(self, gate):
        if self.state != Qubit.KET:
            raise ValueError("State must be a Ket")
        return self.__new(gate.matrix * self.vector)

    def to_register(qubits):
        return reduce(lambda acc, q: acc.tensor(q), qubits)

    def inner(self, other):
        if self.state != Qubit.KET and other.state != Qubit.KET:
            raise ValueError("Both qubits must be kets")
        return (self.conjugate().vector * other.vector)[0, 0]

    def outer(self, other):
        if self.state != Qubit.KET and other.state != Qubit.KET:
            raise ValueError("Both qubits must be kets")
        return self.vector * other.conjugate().vector 

    def __add__(self, other):
        return self.__operation(np.add, other)

    def __sub__(self, other):
        return self.__operation(np.subtract, other)

    def __neg__(self):
        return self.__new(-self.vector, self.state)

    def __mul__(self, other):
        if isinstance(other, Qubit):
            if self.state != Qubit.KET or other.state != Qubit.KET:
                raise ValueError("Both qubits have to be kets")
            return self.tensor(other)
        elif isinstance(other, int) or isinstance(other, float):
            return self.__new(other * self.vector, state = Qubit.KET)
        else:
            raise ValueError("* Qubit undefined for " + str(type(other)))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __operation(self, operation, other):
        if self.state != other.state:
            raise ValueError("Both qubits must be on the same state")
        return self.__new(operation(self.vector, other.vector), self.state)

    def __repr__(self):
        v = self.vector if self.state == Qubit.BRA else np.transpose(self.vector)
        return repr(v)

q0 = Qubit(0)
q1 = Qubit(1)
