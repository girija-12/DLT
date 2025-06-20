def step_function(x):
    return 1 if x >= 1 else 0

def mcp_neuron(inputs, weights, bias):
    weighted_sum = sum(i * w for i, w in zip(inputs, weights)) + bias
    return step_function(weighted_sum)

def XOR(x1, x2):
    return x1 ^ x2 

def logic_gate(gate_type):
    print(f"\n{gate_type} Gate:")
    if gate_type == "AND":
        weights = [1, 1]
        bias = -1.5
        inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]

    elif gate_type == "OR":
        weights = [1, 1]
        bias = -0.5
        inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]

    elif gate_type == "NOR":
        weights = [-1, -1]
        bias = 0.5
        inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]

    elif gate_type == "NOT":
        weights = [-1]
        bias = 0.5
        inputs = [(0,), (1,)]

    elif gate_type == "XOR":
        inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
        for inp in inputs:
            output = XOR(inp[0], inp[1])
            print(f"Input: {inp} => Output: {output}")
        return

    else:
        print("Invalid gate type.")
        return

    for inp in inputs:
        output = mcp_neuron(inp, weights, bias)
        print(f"Input: {inp} => Output: {output}")

for gate in ["AND", "OR", "NOT", "NOR", "XOR"]:
    logic_gate(gate)