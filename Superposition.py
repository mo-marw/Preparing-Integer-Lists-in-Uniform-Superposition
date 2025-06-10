import math
import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler
from qiskit.circuit.library import RYGate
from qiskit.quantum_info import Statevector
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
import itertools
# OR Tools:
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

from collections import deque
import time
from random import randint

# Version 2



def dec2bin(dez, signed): # takes int and returns string
    if not signed or dez >= 0:
        result = ''
        while dez > 0:
            result = str(dez % 2) + result
            dez //= 2        # divide with no remainder
        return (amount_of_qubits-len(result))*'0'+result # extended to the full number of qubits using leading zeros
    else:
        dez += 2**(amount_of_qubits-1)
        a = dec2bin(dez, signed)
        return '1'+a[1:]

def bin2dec(bin, signed): # takes string and returns int
    power = 1
    result = 0
    for i in range(len(bin)):
        result += power*int(bin[-(i+1)])
        power *= 2
    if signed:
        result -= power*int(bin[-(i+1)])    # if signed then power is now the double of the highest digit and therefore the correct number is substracted
    return result

def hamming_distance(bin1, bin2):   # takes two binary numbers represented as strings
    bit_flips = 0
    for i in range(amount_of_qubits):
        if(bin1[-(i+1)] != bin2[-(i+1)]):
            bit_flips += 1
    return bit_flips            

def highest_hamming_to_all(binlist):  # takes list and returns the greatest minimal hamming distance to any other number and a list of all those fulfilling it
    highest_minimal_hamming = 0
    result = []
    for pos1 in range(len(binlist)):
        lowest_hamming_pos1 = amount_of_qubits       # finds out shortest hamming distance from list[pos1] to any other
        for pos2 in range(len(binlist)):
            current_hamming = hamming_distance(binlist[pos1], binlist[pos2])
            if current_hamming < lowest_hamming_pos1 and not current_hamming == 0:
                lowest_hamming_pos1 = current_hamming
        if lowest_hamming_pos1 == highest_minimal_hamming:
            result.append(binlist[pos1])

        if lowest_hamming_pos1 > highest_minimal_hamming:
            highest_minimal_hamming = lowest_hamming_pos1
            result = [binlist[pos1]]
    return result

# ---------------------------------------------------------------------------------------------------------
# naiver Algorithmus durch einzelne bit flips

def class_bit_flips(binary1, binary2):    # list of bit flips, to get from binary1 to binary2, both binary strings
    bit_flips = []
    for i in range(amount_of_qubits):
        if(binary1[-(i+1)] != binary2[-(i+1)]):
            bit_flips.append(i)
    return bit_flips                     # returns list of positions that need to be flipped, starting with pos 0

def superpos(binlist):              # finds out gates to create superposition between elements of list, gets binary
    
    qc = QuantumCircuit(amount_of_qubits+len(binlist)-1)
    for flip in class_bit_flips(amount_of_qubits*'0', binlist[0]):
        qc.x(flip)
    
    # del list[0]

    for list_pos in range(len(binlist)-1):                         # damit die Liste wie in Lennarts Mail beschrieben ist, müsste die eigehende Liste erst Elemente ab Position 1 haben, da Position 0 vom Anfangszustand |0> abgedeckt wäre
        angle = 2*np.arccos(1/np.sqrt(len(binlist)-list_pos))      # +1 in sqrt laut Lennarts Mail, da i = 0 eigentlich schon oben als erster Flip abgedeckt ist. Hier starten wir allerdings erneut für i = 0, daher ok so
        controlled_ry =  RYGate(angle).control(amount_of_qubits, None, binlist[list_pos])     # die Kontrolle für das RYGate ist, dass sich das Hauptregister im Zustand a_i befindet
        qc.append(controlled_ry, [i for i in range(amount_of_qubits)]+[amount_of_qubits + list_pos])    # richtig so, dass ry kontrolliert ist, von gesamten anderen Zuständen?
        # qc.ry(angle, amount_of_qubits + list_pos)
        for flip in class_bit_flips(binlist[list_pos], binlist[list_pos+1]):
            qc.cx(amount_of_qubits + list_pos, flip)
    # qc_measured = qc.measure_all(inplace=False)
    # sampler = StatevectorSampler()
    # shots = 1000
    # job = sampler.run([qc_measured], shots=shots)
    # result = job.result()
    # print(f" > Counts: {result[0].data.meas.get_counts()}")
    
    print(qc)

    # qc.draw(output='mpl')
    # plt.show()

    # qc.remove_final_measurements()  # no measurements allowed
    statevector = Statevector(qc)
    return statevector

    


def count_simulation(bin_list): # takes binary input list, returns array [gate_count, cycle_count]. Simulation for superpos
    gate_count = len(bin_list)-1 # for the controlled ry gates
    cycle_count = len(bin_list) # of that 1 for the starting gates to create the first state and the rest len(list)-1 for the controlled ry gates
    for i in bin_list[0]:
        gate_count += int(i)
    for pos in range(len(bin_list)-1):
        number1 = bin_list[pos]
        number2 = bin_list[pos+1]
        for i in range(amount_of_qubits):
            if(number1[i] != number2[i]):
                gate_count += 1
                cycle_count += 1
                
    return [gate_count, cycle_count]

"""
def minimal_counts(lst): # takes binary input list and returns minimal gate and cycle counts for superpos
    best_gc_order = np.array([])
    best_cc_order = np.array([])
    
    best_gc_order = np.append(best_gc_order,lst) # gate counts
    best_cc_order = np.append(best_cc_order,lst) # cycle counts
    minimal_gc, minimal_cc = count_simulation(lst)
    for permutation in itertools.permutations(lst, len(lst)):
        current_gc, current_cc = count_simulation(permutation)

        if(current_gc == minimal_gc):
            best_gc_order = np.append(best_gc_order,permutation)
        if(current_cc == minimal_cc):
            best_cc_order = np.append(best_cc_order,permutation)

        if(current_gc < minimal_gc):
            best_gc_order = np.array([permutation])
            minimal_gc = current_gc
        if(current_cc < minimal_cc):
            best_cc_order = np.array([permutation])
            minimal_cc = current_cc 
    best_gc_order = list(best_gc_order)                     #
    for i in range(len(best_gc_order)):                     #
        best_gc_order[i] = str(best_gc_order[i])            # all of this only so that the output is in the regular list format
    best_cc_order = list(best_cc_order)                     #
    for i in range(len(best_cc_order)):                     #
        best_cc_order[i] = str(best_cc_order[i])            #
    return [best_gc_order, best_cc_order, minimal_gc, minimal_cc]
"""


def minimal_counts(bin_list): # takes binary input list and returns minimal gate and cycle counts for superpos
    best_gc_order = bin_list # gate counts
    best_cc_order = bin_list # cycle counts
    minimal_gc, minimal_cc = count_simulation(bin_list)
    for permutation in itertools.permutations(bin_list, len(bin_list)):
        # count_sim nur 1x und min() benutzen damit nur eine zeile
        if(count_simulation(permutation)[0] < minimal_gc):
            best_gc_order = permutation
            minimal_gc = count_simulation(permutation)[0]
        if(count_simulation(permutation)[1] < minimal_cc):
            best_cc_order = permutation
            minimal_cc = count_simulation(permutation)[1] 
    best_gc_order = list(best_gc_order)                     #
    for i in range(len(best_gc_order)):                     #
        best_gc_order[i] = str(best_gc_order[i])            # all of this only so that the output is in the regular list format
    best_cc_order = list(best_cc_order)                     #
    for i in range(len(best_cc_order)):                     #
        best_cc_order[i] = str(best_cc_order[i])            #
    return [best_gc_order, best_cc_order, minimal_gc, minimal_cc]


# ways to find approximately optimal order of list for superpos
# 1. greedy
def greedy_smallest_bitflip(list):
    least_gates = amount_of_qubits*len(list)
    least_cycles = amount_of_qubits*len(list)
    best_greedy_list = list
    for first_pos in range(len(list)):      # each element of the list is tried as the first element, greedy list with least counts is saved
        temp_list = list
        greedy_list = np.array([])
        """
        most_gates = 0
        for pos in range(len(list)):
            ones = list[pos].count('1')
            if(ones > most_gates):
                most_gates = ones
                first_pos = pos         # position of first element = element with most bitfips aka most '1'
        """
        # first_pos = 0
        greedy_list = np.append(greedy_list, temp_list[first_pos])
        temp_list = np.delete(list, first_pos)
        while(len(temp_list) > 0):
            least_bit_flips = amount_of_qubits
            next_pos = 0
            for pos in range(len(temp_list)):
                bit_flips = hamming_distance(greedy_list[-1], temp_list[pos])
                if(bit_flips < least_bit_flips):
                    next_pos = pos
                    least_bit_flips = bit_flips
            greedy_list = np.append(greedy_list, temp_list[next_pos])
            temp_list = np.delete(temp_list, next_pos)
        temp_gate_count = count_simulation(greedy_list)[0]
        if(temp_gate_count < least_gates):
            least_gates = temp_gate_count
            best_greedy_list = greedy_list

    print('Reihenfolge: ', end='')
    for i in range(len(best_greedy_list)):
        print(bin2dec(best_greedy_list[i], signed), end=' ')
    print()
    gates, cycles = count_simulation(best_greedy_list)
    print("Mit dieser Reihenfolge hast du ", gates, " gates und ", cycles, "cycles.")

    # superpos(best_greedy_list)

# improved greedy with shorter runtime, but a little worse results
def greedy_smallest_bit_flip2(binlist):

    most_ones = 0
    candidates = highest_hamming_to_all(binlist)
    first = 0
    starting_zero = '0'*len(binlist[0])
    for i in candidates:
        current_ones = hamming_distance(i, starting_zero)
        if current_ones > most_ones:
            most_ones = current_ones
            first = i

    greedy_list = np.array([])
    greedy_list = np.append(greedy_list, first)
    remove = 0
    for i in range(len(binlist)):
        if binlist[i] == first:
            remove = i
    binlist = np.delete(binlist, remove)
    """
    most_gates = 0
    for pos in range(len(list)):
        ones = list[pos].count('1')
        if(ones > most_gates):
            most_gates = ones
            first_pos = pos         # position of first element = element with most bitfips aka most '1'
    """
    # first_pos = 0
    while(len(binlist) > 0):
        least_bit_flips = amount_of_qubits
        next_pos = 0
        for pos in range(len(binlist)):
            bit_flips = hamming_distance(greedy_list[-1], binlist[pos])
            if(bit_flips < least_bit_flips):
                next_pos = pos
                least_bit_flips = bit_flips
        greedy_list = np.append(greedy_list, binlist[next_pos])
        binlist = np.delete(binlist, next_pos)

    print('Reihenfolge: ', end='')
    for i in range(len(greedy_list)):
        print(bin2dec(greedy_list[i], signed), end=' ')
    print()
    gates, cycles = count_simulation(greedy_list)
    print("Mit dieser Reihenfolge hast du ", gates, " gates und ", cycles, "cycles.")

    # superpos(best_greedy_list)
    print("length: ", len(greedy_list))
    # return [gates, cycles]
    return greedy_list

# 2. shortest hamiltonian path (we use tsp and ignore the last step back to the beginning)
# Googles OR-Tool:
def create_data_model(binlist, min_cycles): # gets binary list, boolean min_cycles
    
    data = {}

    data["distance_matrix"] = []
    for i in binlist:
        row = []
        for j in binlist:
            row.append(hamming_distance(i,j))
        data["distance_matrix"].append(row)

    for i in range(len(binlist)):
        data["distance_matrix"][i][0] = 0

    data["num_vehicles"] = 1
    data["depot"] = 0

    if min_cycles:
        for i in range(1, len(binlist)):
            data["distance_matrix"][0][i] = 1
    return data

def print_solution(manager, routing, solution):
    """Prints solution on console."""
    # print(f"Objective: {solution.ObjectiveValue()} miles")
    index = routing.Start(0)
    plan_output = "Route for vehicle 0:\n"
    route_distance = 0
    node_list = []
    while not routing.IsEnd(index):
        plan_output += f" {manager.IndexToNode(index)} ->"
        node_list.append(manager.IndexToNode(index))
        previous_index = index
        index = solution.Value(routing.NextVar(index))
        route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
    plan_output += f" {manager.IndexToNode(index)}\n"
    # print(plan_output)
    plan_output += f"Route distance: {route_distance}miles\n"
    return node_list

def distance_callback(from_index, to_index):
    """Returns the distance between the two nodes."""
    # Convert from routing variable Index to distance matrix NodeIndex.
    from_node = manager.IndexToNode(from_index)
    to_node = manager.IndexToNode(to_index)
    return data["distance_matrix"][from_node][to_node]

def tsp_solver(binlist, min_cycles):    # if min_cycles = True: try to reduce cycle count. if min_cycles = False: try to reduce gate count

    starting_zero = np.array(['0'*len(binlist[0])])
    binlist = np.concatenate((starting_zero, binlist))
    
    """Entry point of the program."""
    # Instantiate the data problem.
    global data 
    data = create_data_model(binlist, min_cycles)

    # Create the routing index manager.
    global manager
    manager = pywrapcp.RoutingIndexManager(len(data["distance_matrix"]), data["num_vehicles"], data["depot"])

    # Create Routing Model.
    # global routing
    routing = pywrapcp.RoutingModel(manager)

    # global transit_callback_index
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Setting first solution heuristic.
    # global search_parameters
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    # Solve the problem.
    # global solution
    solution = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    tsp_index_list = []
    if solution:
        tsp_index_list = print_solution(manager, routing, solution)

    tsp_binary_list = binlist[tsp_index_list]

    best_tsp = np.delete(tsp_binary_list, 0)
    
    """
    print('Reihenfolge: ', end='')
    for i in range(len(best_tsp)):
        print(bin2dec(best_tsp[i], signed), end=' ')
    """
    print()
    
    gates, cycles = count_simulation(best_tsp)
    print("Mit dieser Reihenfolge hast du ", gates, " gates und ", cycles, "cycles.")
    return [gates, cycles]

    # superpos(best_tsp)


def tsp_solver_guided_local_search(binlist, min_cycles, duration):    # if min_cycles = True: try to reduce cycle count. if min_cycles = False: try to reduce gate count

    starting_zero = np.array(['0'*len(binlist[0])])
    binlist = np.concatenate((starting_zero, binlist))
    

    # rotate binary_states, so that the list starts with each number once so we can check which starting point creates the best tsp solution

    
    """Entry point of the program."""
    # Instantiate the data problem.
    global data 
    data = create_data_model(binlist, min_cycles)

    # Create the routing index manager.
    global manager
    manager = pywrapcp.RoutingIndexManager(len(data["distance_matrix"]), data["num_vehicles"], data["depot"])

    # Create Routing Model.
    # global routing
    routing = pywrapcp.RoutingModel(manager)

    # global transit_callback_index
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Setting first solution heuristic.
    # global search_parameters
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.local_search_metaheuristic = (routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search_parameters.time_limit.seconds = duration
    search_parameters.log_search = True

    # Solve the problem.
    # global solution
    solution = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    tsp_index_list = []
    if solution:
        tsp_index_list = print_solution(manager, routing, solution)

    tsp_binary_list = binlist[tsp_index_list]

    best_tsp = np.delete(tsp_binary_list, 0)
    """    
    print('Reihenfolge: ', end='')
    for i in range(len(best_tsp)):
        print(bin2dec(best_tsp[i], signed), end=' ')
    print()
    """
    gates, cycles = count_simulation(best_tsp)
    print("Mit dieser Reihenfolge hast du ", gates, " gates und ", cycles, "cycles.")

    # superpos(best_tsp)
    return [gates, cycles]



# ---------------------------------------------------------------------------------------------------------
#Zwei Zahlen hintereinander, erste davon an gerader Position

# kann für zwei Zahlen auch einfach die erste Zahl in binär sein. Bei allen Einsen wird dann ein X-Gate benötigt.

def two_successive_numbers_first_even(list): #benötigt aktuell die Liste in Dezimaldarstellung
    qc = QuantumCircuit(amount_of_qubits)

    # gates = np.full((amount_of_qubits), '-') #am Anfang keine Gates, alle mit '-' gekennzeichnet

    

    # if(len(list) == 2 and list[0]%2 == 0 and superpos[0]+1 == superpos[1]):
    # gates[0] = 'H'
    qc.h(0)
    temp_gate = amount_of_qubits-1 #höchstes Gate
    temp_pos = list[0] #immer halbieren und gucken, ob ein X gate benötigt wird
    for i in range(amount_of_qubits):
        if(temp_pos-2**(temp_gate) >= 0):
            # gates[temp_gate] = 'X'
            qc.x(temp_gate)
            temp_pos -= 2**(temp_gate)
        temp_gate -= 1
    # print(gates)

    qc_measured = qc.measure_all(inplace=False)
    sampler = StatevectorSampler()
    shots = 1000
    job = sampler.run([qc_measured], shots=shots)
    result = job.result()
    print(f" > Counts: {result[0].data.meas.get_counts()}")
    print(qc)

    #qc.draw(output='mpl')
    # plt.show()

    qc.remove_final_measurements()  # no measurements allowed
    statevector = Statevector(qc)
    print(statevector)

# ---------------------------------------------------------------------------------------------------------

def statevector_interpreter(statevector):
    for i in range(len(statevector)):
        if(statevector[i].real > 0.0001):   # only real part needed, we never have an imaginary part. if too small, it is just an error
            result = i%(2**amount_of_qubits)    # overall state (incl. ancilla register)
            result = dec2bin(result, 0)         # this way, if we have no signed numbers, the numbers are converted into binary and back with no consequence
            result = bin2dec(result, signed)    # but if it should have a sign, it is regained this way
            print("State ", result, " with probability of ", round(statevector[i].real**2*100, 1), "%")



# Input:
# Input by hand

args = sys.argv

if not len(args) == 1:
    input = [int(sys.argv[i]) for i in range(1, len(sys.argv))]
else:
    temp = input("Bitte Zahlen ").split()
    input = [int(temp[i]) for i in range(len(temp))]

"""
# Input by random numbers
negatives = True
input = [randint(1, 10000) for p in range(0, 1000)]
if negatives:
    for pos in range(len(input)):
        if randint(1,2) == 1:
            input[pos] *= -1
# print(input)
"""

# find and erase doubles and decide if signed (with negative numbers) or not:

signed = 0
temp_list = []
for entry in input:
    if entry not in temp_list:
        temp_list.append(entry)
    if entry < 0:
        signed = 1


input = temp_list

abs_input = [abs(value) for value in input]
amount_of_states = len(input)
amount_of_qubits = math.ceil(math.log(max(abs_input)+1)/math.log(2)) + signed   # does not include ancilla qubits

binary_states = np.array([])
for element in input:
    binary_states = np.append(binary_states, dec2bin(element, signed))  # list of strings
print()
# print("desired decimal states:", input)
# print("desired binary states: ", binary_states)

print("Mit dieser Reihenfolge hast du ", count_simulation(binary_states)[0], " gates und ", count_simulation(binary_states)[1], "cycles.")

print(binary_states)

# statevector_interpreter(superpos(binary_states))

a = tsp_solver_guided_local_search(binary_states, False, 1)[0]
b = tsp_solver_guided_local_search(binary_states, True, 1)[1]

print("tsp solver gates: ", a, " and cycles: ", b)


minimal_gates, minimal_cycles, minimal_gate_count, minimal_cycle_count = minimal_counts(binary_states)
minimal_gates_dez = np.array([bin2dec(minimal_gates[0][i], signed) for i in range(len(minimal_gates[0]))])
minimal_cycles_dez = np.array([bin2dec(minimal_cycles[0][i], signed) for i in range(len(minimal_cycles[0]))])

print(minimal_gates)

print("Eine optimale Reihenfolge für gates ist ", minimal_gates_dez, " mit ", minimal_gate_count, " gates.")
print("Eine optimale Reihenfolge für cycles ist ", minimal_cycles_dez, " mit ", minimal_cycle_count, " cycles.")
print()

print("Greedy: ", count_simulation(greedy_smallest_bit_flip2(binary_states)))



"""
# which function is best suited for solving the problem?:
if(len(input) == 2 and input[0]%2 == 0 and input[0]+1 == input[1]):
    two_successive_numbers_first_even(input)
else:
    superpos(binary_states)
"""

# choice = minimal_cycles

# print("\nMinimal Cycles:")
# statevector_interpreter(superpos(choice))
# superpos(choice)

"""
time_start = time.time()
print("\nTSP new, min gates:")
tsp_min_gates1 = tsp_solver_guided_local_search(binary_states, False, 20)
time_stop = time.time()
print(time_stop-time_start)

time_start = time.time()
print("\nTSP, min gates:")
tsp_min_gates_old = tsp_solver(binary_states, False)
time_stop = time.time()
print(time_stop-time_start)

time_start = time.time()
print("\nnew greedy:")
greedy1 = greedy_smallest_bit_flip2(binary_states)
time_stop = time.time()
print(time_stop-time_start)

print(tsp_min_gates1)
# print(tsp_min_gates2)

# print(tsp_min_cycles1)
# print(tsp_min_gates2)
# print(tsp_min_cycles2)
print(tsp_min_gates_old)

print(greedy1)
"""

# Algorithmus auch mal nur 2,3,4,5 Sekunden laufen lassen und gucken ob dann guided local search auch schon verbesserung zu old macht

"""
time_start = time.time()
print("\nGreedy:")
# statevector_interpreter(superpos(greedy_smallest_bitflip(binary_states)))
greedy_smallest_bitflip(binary_states)
time_stop = time.time()
print(time_stop-time_start)
# print()
"""

# als erstes die zahl mit meisten bit flips, damit cycle count niedrig -> in overleaf dokumentieren dass bit flips parallelisierbar
# evtl argument schreiben, warum tsp es gut annähert, obwohl end-besuch vom anfang alles ja verändert


# main einführen, um ordentlicher zu machen. generell aufräumen
# chatgpt nach verbesserungen fragen in bezug auf ordnung
# dez überall in dec umbenennen?
# überall kennzeichnen ob eingegebene Liste binär oder dezimal indem benennung: binlist/ declist ?
# es darf eh nirgendwo list heißen wegen der gleichnamigen python funktion

# verschnellerung: gates/cycles nicht in extra funktion ausrechnen, sondern über objective.
# ancillas über qiskit "richtig" mit einbauen?