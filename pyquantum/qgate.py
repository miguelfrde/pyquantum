import numpy as np
from functools import reduce
from pyquantum.qubit import q0, q1

# Quantum gate representation, contains a the gate matrix and a string
# representation of the gate
class QGate:
    
    # Create a new gate for a given matrix with a given name
    def __init__(self, matrix, name):
        self.matrix = matrix
        self.name = name

    # Performs the tensor product between the gate and another gate g
    def tensor(self, g):
        return QGate(np.kron(self.matrix, g.matrix), str(self) + "(x)" + str(g))
    
    # Applies the gate to a qubit or a qubit register q
    def of(self, q):
        return q.apply_gate(self)

    # Creates a new gate by performing the tensor product of all gates from
    # a given list
    @staticmethod
    def tensor_of_gates(gates):
        return reduce(lambda acc, g: acc.tensor(g), gates)
    
    # Performs and operation with string representation s between the gate and
    # another gate g
    def __operation(self, g, operation, s):
        return QGate(operation(self.matrix, g.matrix), str(self) + s + str(g))

    # Adds two quantum gates (matrix addition)
    def __add__(self, other):
        return self.__operation(other, np.add, "+")

    # Subtracts two quantum gates (matrix subtraction)
    def __sub__(self, other):
        return self.__operation(other, np.subtract, "-")

    # Multiplies two quantum gates (matrix multiplication)
    def __mul__(self, other):
        if not isinstance(other, QGate):
            return QGate(other * self.matrix, str(other) + "*" + str(self))
        return self.__operation(other, np.dot, " * ")

    # String representation of the gate
    def __repr__(self):
        return self.name

    __rmul__ = __mul__
    __str__ = __repr__

# Pauli Gates
X = QGate([[0, 1], [1, 0]], "X")
Z = QGate([[1, 0], [0, -1]], "Z")
I = QGate([[1, 0], [0, 1]], "I")

# Hadamard transform gate
H = QGate([[1/np.sqrt(2), 1/np.sqrt(2)], [1/np.sqrt(2), -1/np.sqrt(2)]], "H")

# Projection gates
P0 = QGate(q0.outer(q0), "P0")
P1 = QGate(q1.outer(q1), "P1")

