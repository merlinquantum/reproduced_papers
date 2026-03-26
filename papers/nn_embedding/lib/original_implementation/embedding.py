import pennylane as qml
from pennylane import numpy as np

N_layers = 3

# exp(ixZ) gate
def exp_Z(x, wires, inverse=False):
  if inverse == False:
    qml.RZ(-2 * x, wires=wires)
  elif inverse == True:
    qml.RZ(2 * x, wires=wires)

# exp(ixZZ) gate
def exp_ZZ1(x, wires, inverse=False):
  if inverse == False:
    qml.CNOT(wires=wires)
    qml.RZ(-2 * x, wires=wires[1])
    qml.CNOT(wires=wires)
  elif inverse == True:
    qml.CNOT(wires=wires)
    qml.RZ(2 * x, wires=wires[1])
    qml.CNOT(wires=wires)

# exp(i(pi - x1)(pi - x2)ZZ) gate
def exp_ZZ2(x1, x2, wires, inverse=False):
  if inverse == False:
    qml.CNOT(wires=wires)
    qml.RZ(-2 * (np.pi - x1) * (np.pi - x2), wires=wires[1])
    qml.CNOT(wires=wires)
  elif inverse == True:
    qml.CNOT(wires=wires)
    qml.RZ(2 * (np.pi - x1) * (np.pi - x2), wires=wires[1])
    qml.CNOT(wires=wires)


# Quantum Embedding 1 for model 1 (Conventional ZZ feature embedding)
def QuantumEmbedding1(input):
  for i in range(N_layers):
    for j in range(8):
      qml.Hadamard(wires=j)
      exp_Z(input[j], wires=j)
    for k in range(7):
      exp_ZZ2(input[k], input[k+1], wires=[k,k+1])
    exp_ZZ2(input[7], input[0], wires=[7, 0])

def QuantumEmbedding1_inverse(input):
  for i in range(N_layers):
    exp_ZZ2(input[7], input[0], wires=[7, 0], inverse=True)
    for k in reversed(range(7)):
      exp_ZZ2(input[k], input[k+1], wires=[k,k+1], inverse=True)
    qml.Barrier()
    for j in range(8):
      exp_Z(input[j], wires=j, inverse=True)
      qml.Hadamard(wires=j)

# Quantum Embedding 2 for model 2
def QuantumEmbedding2(input):
  for i in range(N_layers):
    for j in range(8):
      qml.Hadamard(wires=j)
      exp_Z(input[j], wires=j)
    for k in range(7):
      exp_ZZ1(input[8+k], wires=[k, k+1])
    exp_ZZ1(input[15], wires=[7,0])

def QuantumEmbedding2_inverse(input):
  for i in range(N_layers):
    exp_ZZ1(input[15], wires=[7,0], inverse=True)
    for k in reversed(range(7)):
      exp_ZZ1(input[k+8], wires=[k,k+1], inverse=True)
    qml.Barrier()
    for j in range(8):
      exp_Z(input[j], wires=j, inverse=True)
      qml.Hadamard(wires=j)


      
# Add 4 qubit embedding for demonstrations
def Four_QuantumEmbedding1(input):
  for i in range(N_layers):
    for j in range(4):
      qml.Hadamard(wires=j)
      exp_Z(input[j], wires=j)
    for k in range(3):
      exp_ZZ2(input[k], input[k+1], wires=[k,k+1])
    exp_ZZ2(input[3], input[0], wires=[3, 0])

def Four_QuantumEmbedding1_inverse(input):
  for i in range(N_layers):
    exp_ZZ2(input[3], input[0], wires=[3, 0], inverse=True)
    for k in reversed(range(3)):
      exp_ZZ2(input[k], input[k+1], wires=[k,k+1], inverse=True)
    qml.Barrier()
    for j in range(4):
      exp_Z(input[j], wires=j, inverse=True)
      qml.Hadamard(wires=j)

def Four_QuantumEmbedding2(input):
  for i in range(N_layers):
    for j in range(4):
      qml.Hadamard(wires=j)
      exp_Z(input[j], wires=j)
    for k in range(3):
      exp_ZZ1(input[4 + k], wires=[k,k+1])
    exp_ZZ1(input[7], wires=[3, 0])

def Four_QuantumEmbedding2_inverse(input):
  for i in range(N_layers):
    exp_ZZ1(input[7], wires=[3, 0], inverse=True)
    for k in reversed(range(3)):
      exp_ZZ1(input[k+4], wires=[k,k+1], inverse=True)
    qml.Barrier()
    for j in range(4):
      exp_Z(input[j], wires=j, inverse=True)
      qml.Hadamard(wires=j)


# Noisy 4 qubit embedding
"""
def Noisy_Four_QuantumEmbedding1(input):
  def transform(x):
    if x == 0:
      X = 13
    elif x == 1:
      X = 12
    elif x == 2:
      X = 15
    elif x == 3:
      X = 18
    return X
  
  for j in range(4):
    qml.Hadamard(wires=transform(j))
    exp_Z(input[j], wires=transform(j))
  for k in range(3):
    exp_ZZ2(input[k], input[k+1], wires=[transform(k),transform(k+1)])

"""
def Noisy_Four_QuantumEmbedding1(input):
  for j in range(4):
    qml.Hadamard(wires=j)
    exp_Z(input[j], wires=j)
  for k in range(3):
    exp_ZZ2(input[k], input[k+1], wires=[k,k+1])



def Noisy_Four_QuantumEmbedding1_inverse(input):
  for k in reversed(range(3)):
    exp_ZZ2(input[k], input[k+1], wires=[k,k+1], inverse=True)
  for j in range(4):
    exp_Z(input[j], wires=j, inverse=True)
    qml.Hadamard(wires=j)

def Noisy_Four_QuantumEmbedding2(input):
  for j in range(4):
    qml.Hadamard(wires=j)
    exp_Z(input[j], wires=j)
  for k in range(3):
    exp_ZZ1(input[4 + k], wires=[k,k+1])
  
"""
def Noisy_Four_QuantumEmbedding2(input):
  def transform(x):
    if x == 0:
      X = 13
    elif x == 1:
      X = 12
    elif x == 2:
      X = 15
    elif x == 3:
      X = 18
    return X
  
  for j in range(4):
    qml.Hadamard(wires=transform(j))
    exp_Z(input[j], wires=transform(j))
  for k in range(3):
    exp_ZZ1(input[4 + k], wires=[transform(k),transform(k+1)])
"""
def Noisy_Four_QuantumEmbedding2_inverse(input):
  for k in reversed(range(3)):
    exp_ZZ1(input[k+4], wires=[k,k+1], inverse=True)
  for j in range(4):
    exp_Z(input[j], wires=j, inverse=True)
    qml.Hadamard(wires=j)


# Add 4 qubit noisy embedding for demonstrations
def U_SU4(params, wires): # 15 params
    qml.U3(params[0], params[1], params[2], wires=wires[0])
    qml.U3(params[3], params[4], params[5], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.RY(params[6], wires=wires[0])
    qml.RZ(params[7], wires=wires[1])
    qml.CNOT(wires=[wires[1], wires[0]])
    qml.RY(params[8], wires=wires[0])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.U3(params[9], params[10], params[11], wires=wires[0])
    qml.U3(params[12], params[13], params[14], wires=wires[1])

def U_TTN(params, wires):  # 2 params
    qml.RY(params[0], wires=wires[0])
    qml.RY(params[1], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])
  
def QCNN_four(params, ansatz):
    if ansatz == 'SU4':
      U = U_SU4
      num_params = 15
    elif ansatz == 'TTN':
      U = U_TTN
      num_params = 2
    
    param1 = params[0:num_params]
    param2 = params[num_params:2 * num_params]
    U(param1, wires=[0, 1])
    U(param1, wires=[2, 3])
    U(param1, wires=[1, 2])
    U(param1, wires=[3, 0])
    U(param2, wires=[0, 2])

def Noisy_QCNN_four(params, ansatz):
    if ansatz == 'SU4':
      U = U_SU4
      num_params = 15
    elif ansatz == 'TTN':
      U = U_TTN
      num_params = 2
    
    param1 = params[0:num_params]
    param2 = params[num_params:2 * num_params]
    U(param1, wires=[0, 1])
    U(param1, wires=[2, 3])
    U(param1, wires=[1, 2])
    #U(param1, wires=[3, 0])
    U(param2, wires=[0, 2])

