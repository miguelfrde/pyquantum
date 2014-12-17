from functools import reduce
import numpy as np

# Qubit representation on a Ket or Bra state. Allows to create only
# basis qubits |0> or |1>. By performing operations defined operations
# it's possible to get other qubits or qubit registers.
class Qubit:
    KET = True
    BRA = False
    
    # Creates a new qubit |n> or <n| where n is one or zero 
    def __init__(self, n, state=KET):
        if state != Qubit.KET and state != Qubit.BRA:
            raise ValueError("State must be either KET or BRA")
        self.vector = np.matrix([[1], [0]] if n == 0 else [[0], [1]])
        self.state = state

    # Private helpler method to create a new qubit or qubit register from a
    # vector
    def __new(self, vector, state=KET):
        q = Qubit(1, state)
        q.vector = vector
        return q

    # Computes the conjugate of a qubit
    def conjugate(self):
        return self.__new(np.transpose(np.conjugate(self.vector)),
                not self.state)

    # Tensor of the qubit with another one
    def tensor(self, other):
        return self.__new(np.kron(self.vector, other.vector), self.state)
   
    # Applies the given gate
    def apply_gate(self, gate):
        if self.state != Qubit.KET:
            raise ValueError("State must be a Ket")
        return self.__new(gate.matrix * self.vector)

    # Performs the tensor product of a given list of qubits to create a
    # qubit register
    def to_register(qubits):
        return reduce(lambda acc, q: acc.tensor(q), qubits)

    # Performs the inner product <self|other> of the qubit with another qubit
    def inner(self, other):
        if self.state != Qubit.KET and other.state != Qubit.KET:
            raise ValueError("Both qubits must be kets")
        return (self.conjugate().vector * other.vector)[0, 0]

    # Performs the outer product |self><other| of the qubit with another qubit
    def outer(self, other):
        if self.state != Qubit.KET and other.state != Qubit.KET:
            raise ValueError("Both qubits must be kets")
        return self.vector * other.conjugate().vector 

    # Adds two qubits
    def __add__(self, other):
        return self.__operation(np.add, other)

    # Subtracts two qubits
    def __sub__(self, other):
        return self.__operation(np.subtract, other)

    # Negates a qubit
    def __neg__(self):
        return self.__new(-self.vector, self.state)

    # Multiplies two qubits. If the argument is an int or a float, it performs
    # a multiplication by a scalar. If it's another qubit, it performs the
    # tensor product
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

    # Private method that applies the given operation between the qubit and the
    # other qubit
    def __operation(self, operation, other):
        if self.state != other.state:
            raise ValueError("Both qubits must be on the same state")
        return self.__new(operation(self.vector, other.vector), self.state)

    # Vector representation of the qubit
    def __repr__(self):
        v = self.vector if self.state == Qubit.BRA else np.transpose(self.vector)
        return repr(v)

q0 = Qubit(0)
q1 = Qubit(1)
