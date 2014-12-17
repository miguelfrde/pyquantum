import numpy as np
from functools import reduce
from pyquantum.qubit import q0, q1

class QGate:
    def __init__(self, matrix, name):
        self.matrix = matrix
        self.name = name

    def tensor(self, g):
        return QGate(np.kron(self.matrix, g.matrix), str(self) + "(x)" + str(g))
    
    def of(self, q):
        return q.apply_gate(self)

    @staticmethod
    def tensor_of_gates(gates):
        return reduce(lambda acc, g: acc.tensor(g), gates)
    
    def __operation(self, g, operation, s):
        return QGate(operation(self.matrix, g.matrix), str(self) + s + str(g))

    def __add__(self, other):
        return self.__operation(other, np.add, "+")

    def __sub__(self, other):
        return self.__operation(other, np.subtract, "-")

    def __mul__(self, other):
        if not isinstance(other, QGate):
            return QGate(other * self.matrix, str(other) + "*" + str(self))
        return self.__operation(other, np.dot, " * ")

    def __repr__(self):
        return self.name

    __rmul__ = __mul__
    __str__ = __repr__


class ControlledQGate(QGate):
    def __init__(self, index, gate):
        self.index = index
        self.gate = gate
        self.matrix = self.__gen_matrix().matrix
        self.name = "C" + str(gate)

    def __gen_matrix(self):
        def aux(n):
            if n == 1:
                return P0.tensor(I) + P1.tensor(self.gate)
            p = QGate.tensor_of_gates([P0] + [I for _ in range(n)])
            return p + P1.tensor(aux(n-1))
        return aux(self.index)


X = QGate([[0, 1], [1, 0]], "X")
Z = QGate([[1, 0], [0, -1]], "Z")
I = QGate([[1, 0], [0, 1]], "I")
H = QGate([[1/np.sqrt(2), 1/np.sqrt(2)], [1/np.sqrt(2), -1/np.sqrt(2)]], "H")
P0 = QGate(q0.outer(q0), "P0")
P1 = QGate(q1.outer(q1), "P1")

