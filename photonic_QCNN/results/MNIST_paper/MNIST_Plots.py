# ruff: noqa: E402, F405, N801, F403, N999
"""
To plot the training metrics. Code directly from: https://github.com/ptitbroussou/Photonic_Subspace_QML_Toolkit
"""

### Loading the required libraries
import numpy as np
import torch

print("WARNING. Is torch.cuda.is_available():", torch.cuda.is_available())
train_epochs = 20

nbr_test = 5
training_loss, training_acc, testing_loss, testing_acc = [], [], [], []
for test in range(nbr_test):
    (
        training_loss_list,
        training_accuracy_list,
        testing_loss_list,
        testing_accuracy_list,
    ) = torch.load(f"MNIST_training_data_{test}.pt")
    # saving the data to perform expectation and average values:
    training_loss.append(training_loss_list)
    training_acc.append(training_accuracy_list * 100)
    testing_loss.append(testing_loss_list)
    testing_acc.append(testing_accuracy_list * 100)

training_loss, training_acc, testing_loss, testing_acc = (
    np.array(training_loss),
    np.array(training_acc),
    np.array(testing_loss),
    np.array(testing_acc),
)


# Average and Standard deviation calculus:
average_train_loss = np.average(training_loss, axis=0)
average_train_acc = np.average(training_acc, axis=0)
average_test_loss = np.average(testing_loss, axis=0)
average_test_acc = np.average(testing_acc, axis=0)

std_train_loss = np.std(training_loss, axis=0)
std_train_acc = np.std(training_acc, axis=0)
std_test_loss = np.std(testing_loss, axis=0)
std_test_acc = np.std(testing_acc, axis=0)


print("Training data:")
print(average_train_acc[-1], std_train_acc[-1])
print("Test data:")
print(average_test_acc[-1], std_test_acc[-1])


# Plotting the results:
import matplotlib.pyplot as plt

# Front Size
SMALL_SIZE = 12
MEDIUM_SIZE = 13
BIGGER_SIZE = 15

# Set the font size
plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title


# Plotting the results
fig_acc = plt.figure()
plt.plot(
    [i + 1 for i in range(train_epochs)], average_train_acc, label="Train", color="blue"
)
plt.fill_between(
    [i + 1 for i in range(train_epochs)],
    average_train_acc - std_train_acc,
    average_train_acc + std_train_acc,
    color="blue",
    alpha=0.2,
    label="std dev",
)
plt.plot(
    [i + 1 for i in range(train_epochs)], average_test_acc, label="Test", color="orange"
)
plt.fill_between(
    [i + 1 for i in range(train_epochs)],
    average_test_acc - std_test_acc,
    average_train_acc + std_test_acc,
    color="orange",
    alpha=0.2,
    label="std dev",
)
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.savefig("QCNN_Accuracy_MNIST.pdf", dpi=120, format="pdf", bbox_inches="tight")

fig_loss = plt.figure()
plt.plot(
    [i + 1 for i in range(train_epochs)],
    average_train_loss,
    label="Train",
    color="blue",
)
plt.fill_between(
    [i + 1 for i in range(train_epochs)],
    average_train_loss - std_train_loss,
    average_train_loss + std_train_loss,
    color="blue",
    alpha=0.2,
    label="std dev",
)
plt.plot(
    [i + 1 for i in range(train_epochs)],
    average_test_loss,
    label="Test",
    color="orange",
)
plt.fill_between(
    [i + 1 for i in range(train_epochs)],
    average_test_loss - std_test_loss,
    average_test_loss + std_test_loss,
    color="orange",
    alpha=0.2,
    label="std dev",
)
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend()
plt.savefig("QCNN_Loss_MNIST.pdf", dpi=120, format="pdf", bbox_inches="tight")


list_color = [
    "blue",
    "orange",
    "green",
    "red",
    "purple",
    "brown",
    "pink",
    "gray",
    "olive",
    "cyan",
]
# Plotting the results
fig_all_acc = plt.figure()
for test in range(nbr_test):
    plt.plot(
        [i + 1 for i in range(train_epochs)], training_acc[test], color=list_color[test]
    )
    plt.plot(
        [i + 1 for i in range(train_epochs)],
        testing_acc[test],
        color=list_color[test],
        linestyle="--",
    )
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
# plt.legend()
plt.savefig("QCNN_All_Acc_MNIST.pdf", dpi=120, format="pdf", bbox_inches="tight")
