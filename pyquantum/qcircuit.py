from functools import reduce
from pyquantum.qgate import QGate, ControlledQGate, X, I
import numpy as np

class QuantumCircuit:
    def __init__(self, nqubits, steps):
        self.nqubits = nqubits
        self.circuit = np.array([
            [I for j in range(nqubits)] for i in range(steps)])

    def concat(self, other):
        if self.nqubits != other.nqubits:
            raise ValueError("Both circuits must work on the same number of qubits")
        new = QuantumCircuit(self.nqubits, len(self.circuit) + len(other.circuit))
        new.circuit = np.concatenate((self.circuit, other.circuit), axis=0)
        return new

    def add_gate(self, i, j, gate):
        if self.__has_controlled_gate(self.circuit[i, j+1:]): 
            raise ValueError("Cannot apply a gate to a qubit that is controlling")
        self.circuit[i, j] = gate

    def add_controlled_gate(self, i, j, gate):
        if np.any(list(map(lambda g: g != I, self.circuit[i, :j]))):
            raise ValueError("Cannot add a controlled gate where the previous gates are not I")
        self.circuit[i, j] = ControlledQGate(j, gate)
   
    def __has_controlled_gate(self, gates):
        return np.any(list(map(lambda g: isinstance(g, ControlledQGate), gates))) 

    def asgate(self):
        def column_gate(col):
            if self.__has_controlled_gate(col):
                i = next(i for i,g in enumerate(col) if isinstance(g, ControlledQGate))
                return QGate.tensor_of_gates(col[i:])
            return QGate.tensor_of_gates(col)
        gates = map(column_gate, self.circuit)
        return reduce(np.dot, reversed(list(gates)))

#    def __str__(self):
#        # TODO
#        return ""

#    __repr__ = __str__


def __gen_cnot():
    circ = QuantumCircuit(2, 1)
    circ.add_controlled_gate(0, 1, X)
    return circ

def __gen_toffoli():
    circ = QuantumCircuit(3, 1)
    circ.add_controlled_gate(0, 2, X)
    return circ


CNOT = __gen_cnot()
TOFFOLI = __gen_toffoli()
