import pennylane as qml
import torch
from torch import nn
import embedding

dev = qml.device('default.qubit', wires=8)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




"""
This part is a code for Hybrid Model 1.
Hybrid Model 1 transforms 8 dimensional features to 8 dimensional features using Fully connected classical NN.
Model1_Fidelity uses fideliy as a loss function.
Model1_HSinner uses Hilbert-Schmidt inner product as a loss function.
"""
@qml.qnode(dev, interface="torch")
def circuit1(inputs): 
    embedding.QuantumEmbedding1(inputs[0:8])
    embedding.QuantumEmbedding1_inverse(inputs[8:16])
    return qml.probs(wires=range(8))

class Model1_Fidelity(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qlayer1 = qml.qnn.TorchLayer(circuit1, weight_shapes={})
        self.linear_relu_stack1 = nn.Sequential(
            nn.Linear(8, 10),
            nn.ReLU(),
            nn.Linear(10,10),
            nn.ReLU(),
            nn.Linear(10,8)
        )
    def forward(self, x1, x2):
        x1 = self.linear_relu_stack1(x1)
        x2 = self.linear_relu_stack1(x2)
        x = torch.concat([x1, x2], 1)
        x = self.qlayer1(x)
        return x[:,0]

class Model1_HSinner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.matrix_fn1 = qml.matrix(circuit1)
        self.linear_relu_stack1 = nn.Sequential(
            nn.Linear(8, 10),
            nn.ReLU(),
            nn.Linear(10,10),
            nn.ReLU(),
            nn.Linear(10,8)
        )
    def forward(self, x1, x2):
        x1 = self.linear_relu_stack1(x1)
        x2 = self.linear_relu_stack1(x2)
        x = torch.concat([x1, x2], 1).to("cpu")
        x = [torch.real(torch.trace(self.matrix_fn1(a))) for a in x]
        x = torch.stack(x, dim = 0).to(device)
        return x / 2**8




"""
This part is a code for Hybrid Model 2.
Hybrid Model 2 transforms 8 dimensional features to 16 dimensional features.
16 dimensional output is used as a rotation angle of ZZ feature embedding.
Model2_Fidelity uses fidelity loss as a loss function.
Model2_HSinner uses Hilbert-Schmidt inner product as a loss function.
"""
@qml.qnode(dev, interface="torch")
def circuit2(inputs): 
    embedding.QuantumEmbedding2(inputs[0:16])
    embedding.QuantumEmbedding2_inverse(inputs[16:32])
    return qml.probs(wires=range(8))

class Model2_Fidelity(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qlayer2 = qml.qnn.TorchLayer(circuit2, weight_shapes={})
        self.matrix_fn2 = qml.matrix(circuit2)
        self.linear_relu_stack2 = nn.Sequential(
            nn.Linear(8, 20),
            nn.ReLU(),
            nn.Linear(20,20),
            nn.ReLU(),
            nn.Linear(20,16)
        )
    def forward(self, x1, x2):
        x1 = self.linear_relu_stack2(x1)
        x2 = self.linear_relu_stack2(x2)
        x = torch.concat([x1, x2], 1)
        x = self.qlayer2(x)
        return x[:,0]

class Model2_HSinner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qlayer2 = qml.qnn.TorchLayer(circuit2, weight_shapes={})
        self.matrix_fn2 = qml.matrix(circuit2)
        self.linear_relu_stack2 = nn.Sequential(
            nn.Linear(8, 20),
            nn.ReLU(),
            nn.Linear(20,20),
            nn.ReLU(),
            nn.Linear(20,16)
        )
    def forward(self, x1, x2):
        x1 = self.linear_relu_stack2(x1)
        x2 = self.linear_relu_stack2(x2)
        x = torch.concat([x1, x2], 1).to("cpu")
        x = [torch.real(torch.trace(self.matrix_fn2(a))) for a in x]
        x = torch.stack(x, dim=0).to(device)
        return x / 2**8





"""
This part of code implements Hybrid Model 3.
Hybrid Model 3 transforms 28 * 28 dimensional features to 16 dimensional features using CNN.
16 dimensional features are used as a rotation angle of the ZZ feature embedding.
Model3_Fidelity uses fidelity loss as a loss function.
Model3_HSinner uses Hilbert Schmidt inner as a loss function.
"""
@qml.qnode(dev, interface="torch")
def circuit3(inputs): 
    embedding.QuantumEmbedding2(inputs[0:16])
    embedding.QuantumEmbedding2_inverse(inputs[16:32])
    return qml.probs(wires=range(8))

class Model3_Fidelity(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Layer1: 28 * 28 -> 14 * 14
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Layer2: 14 * 14 -> 7 * 7
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Fully connected Layers 7 * 7 -> 16
        self.fc = torch.nn.Linear(7 * 7, 16, bias=True)

        self.qlayer3 = qml.qnn.TorchLayer(circuit3, weight_shapes={})

    def forward(self, x1, x2):
        x1 = self.layer1(x1)
        x1 = self.layer2(x1)
        x1 = x1.view(-1, 7 * 7)
        x1 = self.fc(x1)

        x2 = self.layer1(x2)
        x2 = self.layer2(x2)
        x2 = x2.view(-1, 7 * 7)
        x2 = self.fc(x2)

        x = torch.concat([x1, x2], 1)
        x = self.qlayer3(x)
        return x[:,0]

class Model3_HSinner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.matrix_fn3 = qml.matrix(circuit3)
        # Layer1: 28 * 28 -> 14 * 14
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Layer2: 14 * 14 -> 7 * 7
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Fully connected Layers 7 * 7 -> 16
        self.fc = torch.nn.Linear(7 * 7, 16, bias=True)


    def forward(self, x1, x2):
        x1 = self.layer1(x1)
        x1 = self.layer2(x1)
        x1 = x1.view(-1, 7 * 7)
        x1 = self.fc(x1)

        x2 = self.layer1(x2)
        x2 = self.layer2(x2)
        x2 = x2.view(-1, 7 * 7)
        x2 = self.fc(x2)

        
        x = torch.concat([x1, x2], 1).to("cpu")
        x = [torch.real(torch.trace(self.matrix_fn3(a))) for a in x]
        x = torch.stack(x, dim=0).to(device)
        return x / 2**8




# Training Amplitude Model
class Model_Amplitude(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(16,32),
            nn.ReLU(),
            nn.Linear(32,32),
            nn.ReLU(),
            nn.Linear(32,16)
        )
    def forward(self, x1, x0):
        x1 = self.linear_relu_stack(x1)
        x0 = self.linear_relu_stack(x0)
        return torch.sum(x1 * x0, dim=-1)

"""
Below are hybrid models that uses distance as a loss function.
The codes are out of interest as the are not efficiently calculable with quantum computers.
Use for comparison purposes.
# Hybrid Distance Model1
@qml.qnode(dev, interface="torch")
def distance_circuit1(inputs): 
    embedding.QuantumEmbedding1(inputs[0:8])
    return qml.density_matrix(wires=range(8))

class DistanceModel1_Trace(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qlayer1_distance = qml.qnn.TorchLayer(distance_circuit1, weight_shapes={})
        self.linear_relu_stack1 = nn.Sequential(
            nn.Linear(8, 10),
            nn.ReLU(),
            nn.Linear(10,10),
            nn.ReLU(),
            nn.Linear(10,8)
        )
    def forward(self, x1, x0):
        x1 = self.linear_relu_stack1(x1)
        x0 = self.linear_relu_stack1(x0)
        rhos1 = self.qlayer1_distance(x1)
        rhos0 = self.qlayer1_distance(x0)
        rho1 = torch.sum(rhos1, dim=0) / len(x1)
        rho0 = torch.sum(rhos0, dim=0) / len(x0)
        rho_diff = rho1 - rho0
        eigvals = torch.linalg.eigvals(rho_diff)
        return -0.5 * torch.real(torch.sum(torch.abs(eigvals)))


class DistanceModel1_HS(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qlayer1_distance = qml.qnn.TorchLayer(distance_circuit1, weight_shapes={})
        self.linear_relu_stack1 = nn.Sequential(
            nn.Linear(8, 10),
            nn.ReLU(),
            nn.Linear(10,10),
            nn.ReLU(),
            nn.Linear(10,8)
        )
    def forward(self, x1, x0):
        x1 = self.linear_relu_stack1(x1)
        x0 = self.linear_relu_stack1(x0)
        rhos1 = self.qlayer1_distance(x1)
        rhos0 = self.qlayer1_distance(x0)
        rho1 = torch.sum(rhos1, dim=0) / len(x1)
        rho0 = torch.sum(rhos0, dim=0) / len(x0)
        rho_diff = rho1 - rho0
        return -0.5 * torch.trace(rho_diff @ rho_diff)


# Hybrid Distance Model 2
@qml.qnode(dev, interface="torch")
def distance_circuit2(inputs):
    embedding.QuantumEmbedding2(inputs[0:16])
    return qml.density_matrix(wires=range(8))

class DistanceModel2_Trace(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qlayer2_distance = qml.qnn.TorchLayer(distance_circuit2, weight_shapes={})
        self.linear_relu_stack2 = nn.Sequential(
            nn.Linear(8, 20),
            nn.ReLU(),
            nn.Linear(20,20),
            nn.ReLU(),
            nn.Linear(20,16)
        )
    def forward(self, x1, x0):
        x1 = self.linear_relu_stack2(x1)
        x0 = self.linear_relu_stack2(x0)
        rhos1 = self.qlayer2_distance(x1)
        rhos0 = self.qlayer2_distance(x0)
        rho1 = torch.sum(rhos1, dim=0) / len(x1)
        rho0 = torch.sum(rhos0, dim=0) / len(x0)
        rho_diff = rho1 - rho0
        eigvals = torch.linalg.eigvals(rho_diff)
        return -0.5 * torch.real(torch.sum(torch.abs(eigvals)))


class DistanceModel2_HS(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qlayer2_distance = qml.qnn.TorchLayer(distance_circuit2, weight_shapes={})
        self.linear_relu_stack2 = nn.Sequential(
            nn.Linear(8, 20),
            nn.ReLU(),
            nn.Linear(20,20),
            nn.ReLU(),
            nn.Linear(20,16)
        )
    def forward(self, x1, x0):
        x1 = self.linear_relu_stack2(x1)
        x0 = self.linear_relu_stack2(x0)
        rhos1 = self.qlayer2_distance(x1)
        rhos0 = self.qlayer2_distance(x0)
        rho1 = torch.sum(rhos1, dim=0) / len(x1)
        rho0 = torch.sum(rhos0, dim=0) / len(x0)
        rho_diff = rho1 - rho0
        return -0.5 * torch.trace(rho_diff @ rho_diff)
"""

# Get model function
def get_model(model):
    if model == 'Model1_Fidelity':
        return Model1_Fidelity()
    elif model == 'Model1_HSinner':
        return Model1_HSinner()
    elif model == 'Model2_Fidelity':
        return Model2_Fidelity()
    elif model == 'Model2_HSinner':
        return Model2_HSinner()
    elif model == 'Model3_Fidelity':
        return Model3_Fidelity()
    elif model == 'Model3_HSinner':
        return Model3_HSinner()
    elif model == 'Model_Amplitude':
        return  Model_Amplitude()
    """
    elif model == 'DistanceModel1_Trace':
        return DistanceModel1_Trace()
    elif model == 'DistanceModel1_HS':
        return DistanceModel1_HS()
    elif model == 'DistanceModel2_Trace':
        return DistanceModel2_Trace()
    elif model == 'DistanceModel2_HS':
        return DistanceModel2_HS()
    """
