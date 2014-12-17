from functools import reduce
from pyquantum.qgate import QGate, X, I, P0, P1
import numpy as np

# Used by QuantumCircuit to handle gates controlled by gates on a position
# lower than index
class ControlledQGate(QGate):

    # Creates a new quantum gate on a given index of some column of a
    # quantum circuit
    def __init__(self, index, gate):
        self.index = index
        self.gate = gate
        self.matrix = self.__gen_matrix().matrix
        self.name = "C" + str(gate)

    # Private function used to generate the gate matrix representation
    def __gen_matrix(self):
        def aux(n):
            if n == 1:
                return P0.tensor(I) + P1.tensor(self.gate)
            p = QGate.tensor_of_gates([P0] + [I for _ in range(n)])
            return p + P1.tensor(aux(n-1))
        return aux(self.index)


# Quantum circuit simulation
class QuantumCircuit:
    
    # Initializes a quantum circuit that works on n qubits and consists of
    # a defined number of steps (columns). Initializes it with the Pauli I
    # gate on every position
    def __init__(self, nqubits, steps):
        self.nqubits = nqubits
        self.circuit = np.array([
            [I for j in range(nqubits)] for i in range(steps)])

    # Concatenates the circuit with another one that works on the same number
    # of qubits
    def concat(self, other):
        if self.nqubits != other.nqubits:
            raise ValueError("Both circuits must work on the same number of qubits")
        new = QuantumCircuit(self.nqubits, len(self.circuit) + len(other.circuit))
        new.circuit = np.concatenate((self.circuit, other.circuit), axis=0)
        return new

    # Adds a gate on position i (column), j (row). Fails if there is some
    # controlled gate on the same column on some row with index greater than j
    def add_gate(self, i, j, gate):
        if self.__has_controlled_gate(self.circuit[i, j+1:]): 
            raise ValueError("Cannot apply a gate to a qubit that is controlling")
        self.circuit[i, j] = gate

    # Adds a controlled gate on column i, row j that is controlled by all the
    # previous qubits on the same column
    def add_controlled_gate(self, i, j, gate):
        if np.any(list(map(lambda g: g != I, self.circuit[i, :j]))):
            raise ValueError("Cannot add a controlled gate where the previous gates are not I")
        self.circuit[i, j] = ControlledQGate(j, gate)
   
    # Private method that checks if there is a controllled gate on a list of
    # gates
    def __has_controlled_gate(self, gates):
        return np.any(list(map(lambda g: isinstance(g, ControlledQGate), gates))) 

    # Converts the circuit from an ordered array of gates to a single gate
    def asgate(self):
        def column_gate(col):
            if self.__has_controlled_gate(col):
                i = next(i for i,g in enumerate(col) if isinstance(g, ControlledQGate))
                return QGate.tensor_of_gates(col[i:])
            return QGate.tensor_of_gates(col)
        gates = map(column_gate, self.circuit)
        return reduce(np.dot, reversed(list(gates)))
    
    def __str__(self):
        gates_by_row = np.transpose(self.circuit)
        result = ''
        for r, row in enumerate(gates_by_row):
            result += '---'
            for c, gate in enumerate(row):
                if self.__has_controlled_gate(self.circuit[c, r+1:]):
                    result += 'o---'
                else:
                    g = str(gate)[1:] if isinstance(gate, ControlledQGate) else str(gate)
                    result += g + '-'*(4 - len(g)) if gate != I else '----'
            if r != len(gates_by_row) - 1:
                result += '\n---'
                for c, gate in enumerate(row):
                    if self.__has_controlled_gate(self.circuit[c, r+1:]):
                        result += '|---'
                    else:
                        result += '----'
            result += '\n'
        return result

    __repr__ = __str__


# Private function that creates a CNOT gate
def __gen_cnot():
    circ = QuantumCircuit(2, 1)
    circ.add_controlled_gate(0, 1, X)
    return circ

# Private function that creates a TOFFOLI gate
def __gen_toffoli():
    circ = QuantumCircuit(3, 1)
    circ.add_controlled_gate(0, 2, X)
    return circ

# CNOT gate circuit
CNOT = __gen_cnot()

# TOFFOLI gate circuit
TOFFOLI = __gen_toffoli()
