import numpy as np
from simplenn import SimpleNeuralNetwork


network = SimpleNeuralNetwork()
print(network)

train_inputs = np.array([[1,1,0],[1,1,1],[1,1,0],[1,0,0],[0,1,1],[0,1,0],[0,0,0]])
train_outputs = np.array([[0,1,0,0,1,0,1]]).T
train_itrators = 50_000

network.train(train_inputs,train_outputs,train_itrators)

print(network.weights)

print("ocena modelu")
testdata = np.array([[1,1,1],[1,0,0],[0,1,1],[0,1,0],[0,0,1],[0,0,0],])
for data in testdata:
    print(f"wynik dla {data} -> {network.propagation(data)}")
