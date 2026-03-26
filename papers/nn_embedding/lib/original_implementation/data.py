from os import listdir
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import tensorflow as tf
import torch


def data_load_and_process(dataset, feature_reduction='resize256', classes=[0,1]):
    """
    This part of the code was originally written to use Brain signal dataset.
    This implementation is currently out of interest; hence commented out.
    Will include this back later when needed.

    if dataset == 'signal':
        dataset_signal = pd.read_csv('/data/ROI_' +str(ROI)+ '_df_length256_zero_padding.csv')

        dataset_value = dataset_signal.iloc[:,:-1]
        dataset_label = dataset_signal.iloc[:,-1]

        x_train, x_test, y_train, y_test = train_test_split(dataset_value, dataset_label, test_size=0.2, shuffle=True,
                                                            stratify=dataset_label, random_state=10)

        x_train, x_test, y_train, y_test =\
            x_train.values.tolist(), x_test.values.tolist(), y_train.values.tolist(), y_test.values.tolist()
        y_train = [1 if y == 1 else -1 for y in y_train]
        y_test = [1 if y ==1 else -1 for y in y_test]
    """

    if dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    elif dataset == 'fashion':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train, x_test = x_train[..., np.newaxis] / 255.0, x_test[..., np.newaxis] / 255.0

    if len(classes) == 2:
        train_filter_tf = np.where((y_train == classes[0] ) | (y_train == classes[1] ))
        test_filter_tf = np.where((y_test == classes[0] ) | (y_test == classes[1] ))

    elif len(classes) == 3: # For multicalss classification
        train_filter_tf = np.where((y_train == classes[0] ) | (y_train == classes[1] ) | (y_train == classes[2]))
        test_filter_tf = np.where((y_test == classes[0] ) | (y_test == classes[1] ) | (y_test == classes[2]))  

    x_train, y_train = x_train[train_filter_tf], y_train[train_filter_tf]
    x_test, y_test = x_test[test_filter_tf], y_test[test_filter_tf]

        
    if feature_reduction == False:
        return x_train, x_test, y_train, y_test


    if feature_reduction in ['PCA16', 'PCA8', 'PCA4']:
        x_train = tf.image.resize(x_train[:], (256, 1)).numpy()
        x_test = tf.image.resize(x_test[:], (256, 1)).numpy()
        x_train, x_test = tf.squeeze(x_train).numpy(), tf.squeeze(x_test).numpy()
        if feature_reduction == 'PCA16':
            dim_reduct = 16
        if feature_reduction == 'PCA8':
            dim_reduct = 8
        if feature_reduction == 'PCA4':
            dim_reduct = 4
        X_train = PCA(dim_reduct).fit_transform(x_train)
        X_test = PCA(dim_reduct).fit_transform(x_test)
        x_train, x_test = [], []
        for x in X_train:
            x = (x - x.min()) * (np.pi / (x.max() - x.min()))
            x_train.append(x)
        for x in X_test:
            x = (x - x.min()) * (np.pi / (x.max() - x.min()))
            x_test.append(x)
        return x_train, x_test, y_train, y_test