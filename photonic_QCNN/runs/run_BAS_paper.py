# ruff: noqa: E402, F405, N801, F403, N999
"""
Run file from the paper with slight modifications to enhance performance. Original code is available at:
https://github.com/ptitbroussou/Photonic_Subspace_QML_Toolkit
"""

### Loading the required libraries
import os
import sys

# Add the parent directory of 'photonic_QCNN' to sys.path
# Since runs/ is inside photonic_QCNN/, parent of photonic_QCNN is one level up
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import matplotlib.pyplot as plt
import numpy as np
import torch

# Save path:
# Get the directory where the current script lives
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the folder relative to script_dir
save_dir = os.path.join(script_dir, "..", "results", "BAS_paper")

# Make sure the folder exists
os.makedirs(save_dir, exist_ok=True)

print("WARNING. Is torch.cuda.is_available():", torch.cuda.is_available())

# Fet the absolute path to the root of the Repo (Photonic_Simulation_QCNN):
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
# Add repo root to the system path:
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from photonic_QCNN.data.data import (
    convert_dataset_to_tensor,
    convert_scalar_labels_to_onehot,
    convert_tensor_to_loader,
    get_dataset,
)

# Hyperparameters:s
batch_size = 6
train_dataset_number = 400
test_dataset_number = 200
class_set = [0, 1]
nbr_rows, nbr_cols = 4, 4

# Pennylane Bars and Stripes dataset:
print("Pennylane Bars and Stripes dataset:")
x_train, x_test, y_train, y_test = get_dataset("BAS", "paper", 42)
y_train, y_test = convert_scalar_labels_to_onehot(y_train, y_test)
x_train, x_test, y_train, y_test = convert_dataset_to_tensor(
    x_train, x_test, y_train, y_test
)
train_loader = convert_tensor_to_loader(x_train, y_train, batch_size)
test_loader = convert_tensor_to_loader(x_test, y_test, batch_size)

for data, target in train_loader:
    print("Data shape: ", data.shape)
    print("Target shape: ", target.shape)
    break

# Define the Quantum CNN model:
from photonic_QCNN.src.paper_layers.HW_preserving_QCNN.Conv import (
    Conv_RBS_density_I2,
)
from photonic_QCNN.src.paper_layers.HW_preserving_QCNN.Pooling import (
    Pooling_2D_density_HW,
)
from photonic_QCNN.src.paper_layers.Linear_Optics import *
from photonic_QCNN.src.paper_layers.measurement import *
from photonic_QCNN.src.paper_layers.toolbox_basis_change import (
    Basis_Change_Image_to_larger_Fock_density,
)

### Hyperparameters:
m = nbr_rows + nbr_cols  # number of modes in total
add1, add2 = 1, 1
nbr_class = 2
# list_gates = [(2,3),(1,2),(3,4),(0,1),(2,3),(4,5),(1,2),(3,4),(0,1),(2,3),(4,5),(1,2),(3,4)]
list_gates = [
    (2, 3),
    (1, 2),
    (3, 4),
    (0, 1),
    (2, 3),
    (4, 5),
    (1, 2),
    (3, 4),
]  # New Experimental choice for the dense circuit
modes_detected = [2, 3]


device = torch.device("cpu")


class Photonic_QCNN(nn.Module):
    """A class defining the fully connected neural network"""

    def __init__(self, m, list_gates, modes_detected, device):
        super().__init__()
        self.device = device
        self.Conv = Conv_RBS_density_I2(m // 2, 2, device)
        self.Pool = Pooling_2D_density_HW(m // 2, m // 4, device)
        self.toFock = Basis_Change_Image_to_larger_Fock_density(
            m // 4, m // 4, add1, add2, device
        )
        self.dense = VQC_Fock_BS_density(2, m // 2 + add1 + add2, list_gates, device)
        self.measure = Measure_Photon_detection(
            2, m // 2 + add1 + add2, modes_detected, device
        )
        # self.toFock =  Basis_Change_Image_to_Fock_density(m//4,m//4,device)
        # self.dense = VQC_Fock_BS_density(2, m//2, list_gates,device)
        # self.measure = Measure_Photon_detection(2, m//2, 0, device)

    def forward(self, x):
        x = self.Conv(x)
        x = self.Pool(x)
        return self.measure(self.dense(self.toFock(x)))


### Training the network:
from torch.optim.lr_scheduler import ExponentialLR

from photonic_QCNN.training.training_paper import train_globally_mse

# Define the network:
criterion = torch.nn.MSELoss()
### We run multiple time to have average results and variance:
nbr_test = 5
training_loss, training_acc, testing_loss, testing_acc = [], [], [], []
for test in range(nbr_test):
    print(f"Test number: {test + 1}/{nbr_test}")
    # Define the network:
    network_dense = Photonic_QCNN(m, list_gates, modes_detected, device)
    criterion = torch.nn.MSELoss()

    # optimizer = torch.optim.SGD(network_dense.parameters())
    # optimizer = torch.optim.Adam(network_dense.parameters(), lr=1e-3, weight_decay=1e-4)
    optimizer = torch.optim.Adam(network_dense.parameters(), lr=0.1, weight_decay=0.001)
    scheduler = ExponentialLR(optimizer, gamma=0.9)
    output_scale, train_epochs, test_interval = 1, 20, 1
    # output_scale, train_epochs, test_interval = 66, 20, 1

    (
        network_state,
        training_loss_list,
        training_accuracy_list,
        testing_loss_list,
        testing_accuracy_list,
    ) = train_globally_mse(
        batch_size,
        m // 2,
        network_dense,
        train_loader,
        test_loader,
        optimizer,
        scheduler,
        criterion,
        output_scale,
        train_epochs,
        test_interval,
        device,
    )
    save_path = os.path.join(save_dir, f"model_state_BAS_{test}")
    torch.save(network_state, save_path)  # save network parameters
    training_data = torch.tensor(
        [
            training_loss_list,
            training_accuracy_list,
            testing_loss_list,
            testing_accuracy_list,
        ]
    )
    save_path = os.path.join(save_dir, f"BAS_training_data_{test}.pt")
    torch.save(training_data, save_path)
    # saving the data to perform expectation and average values:
    training_loss.append(training_loss_list)
    training_acc.append(training_accuracy_list)
    testing_loss.append(testing_loss_list)
    testing_acc.append(testing_accuracy_list)


average_train_loss = np.mean(training_loss, axis=0)
std_train_loss = np.std(training_loss, axis=0, ddof=1)  # ddof=1 → N-1

average_train_acc = np.mean(training_acc, axis=0)
std_train_acc = np.std(training_acc, axis=0, ddof=1)

average_test_loss = np.mean(testing_loss, axis=0)
std_test_loss = np.std(testing_loss, axis=0, ddof=1)

average_test_acc = np.mean(testing_acc, axis=0)
std_test_acc = np.std(testing_acc, axis=0, ddof=1)


print(f"Average final train accuracy: {average_train_acc[-1]} ± {std_train_acc[-1]}")
print(f"Average final test accuracy: {average_test_acc[-1]} ± {std_test_acc[-1]}")

epochs = np.arange(1, train_epochs + 1)

plt.figure(figsize=(12, 5))

# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(epochs, average_train_loss, label="Train Loss", color="blue")
plt.fill_between(
    epochs,
    average_train_loss - std_train_loss,
    average_train_loss + std_train_loss,
    color="blue",
    alpha=0.2,
)
plt.plot(epochs, average_test_loss, label="Test Loss", color="red")
plt.fill_between(
    epochs,
    average_test_loss - std_test_loss,
    average_test_loss + std_test_loss,
    color="red",
    alpha=0.2,
)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Test Loss")
plt.legend()
plt.grid(True)

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, average_train_acc, label="Train Accuracy", color="blue")
plt.fill_between(
    epochs,
    average_train_acc - std_train_acc,
    average_train_acc + std_train_acc,
    color="blue",
    alpha=0.2,
)
plt.plot(epochs, average_test_acc, label="Test Accuracy", color="red")
plt.fill_between(
    epochs,
    average_test_acc - std_test_acc,
    average_test_acc + std_test_acc,
    color="red",
    alpha=0.2,
)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Test Accuracy")
plt.legend()
plt.grid(True)
plt.ylim(0, 1)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "BAS_training_metrics.png"))
