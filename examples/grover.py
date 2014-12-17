import numpy as np
import matplotlib.pyplot as plt
import sys
import time

from pyquantum.qubit import q0, q1, Qubit
from pyquantum.qcircuit import QuantumCircuit
from pyquantum.qgate import H, X, Z

# Creates a phase shit circuit for f(x) = 1 where
# x is an n bit number
def gen_oracle(x, n):
    binary = bin(x)[2:]
    binary = '0' * (n - len(binary)) + binary
    circ = QuantumCircuit(n + 1, 3)
    for row, b in enumerate(binary):
        if b == '0':
            circ.add_gate(0, row, X)
            circ.add_gate(2, row, X)
    circ.add_controlled_gate(1, n, X)
    return circ

# Generates an inversion about the mean circuit for n
# qubits
def inv_about_mean(n):
    circ = QuantumCircuit(n+1, 5)
    for row in range(n):
        circ.add_gate(0, row, H)
        circ.add_gate(1, row, X)
        circ.add_gate(3, row, X)
        circ.add_gate(4, row, H)
    circ.add_controlled_gate(2, n-1, Z)
    return circ

# Runs the grover algorithm given an oracle on n qubits
# Plots the result after each iteration
def grover(oracle, n, plot=False):
    initial = Qubit.to_register([q0] * n + [q1])
    circ = QuantumCircuit(n + 1, 1)
    for row in range(n+1):
        circ.add_gate(0, row, H)
    iters = int(np.pi/4 * np.sqrt(2 ** n))
    print('Iterations:', iters) 
    for i in range(iters):
        if plot:
            grover_plot(circ.asgate().of(initial), i, iters)
        circ = circ.concat(oracle).concat(inv_about_mean(n))
    result = circ.asgate().of(initial)
    if plot:
        grover_plot(result, iters, iters)    
    return result

# Helper function to perform the plot given the result at some iteration
# out of a total number of iterations
def grover_plot(result, iteration, total):
    results = [grover_measure(result.vector, x) for x in range(2 ** n)]
    fig = plt.figure()
    plt.plot(list(range(2 ** n)), results,  'o')
    for x, r in enumerate(results):
        plt.plot([x, x], [0, r], 'b-')
    plt.axis([-1, 2 ** n, 0, 1])
    plt.title('Grover iteration ' + str(iteration) + '/' + str(total))
    plt.ylabel('P(x)')
    plt.xlabel('x')
    plt.draw()
    plt.pause(2)
    plt.close(fig)

# Measures the result of a grover result qubit register for a number v
def grover_measure(register, v):
    a = register[2*v] * np.conjugate(register[2*v])
    b = register[2*v + 1] * np.conjugate(register[2*v + 1])
    return (a + b)[0, 0]

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python grover.py x nqubits [--plot]')
        sys.exit(1)
    x0 = int(sys.argv[1])
    n = int(sys.argv[2])
    plot = False
    if len(sys.argv) == 4 and sys.argv[3] == '--plot':
        plot = True
    
    print('f(' + str(x0) + ') = 1')
    print(n, 'qubits')
    
    result = grover(gen_oracle(x0, n), n, plot=plot)
    tot = 0
    for x in range(2 ** n):
        g = grover_measure(result.vector, x)
        print('P(' + str(x) + ') = %.5f' % g)
        tot += g
    print('Sum P(x) =', tot)

