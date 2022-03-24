import numpy as np
from logistic_regression import LogisticRegression
from nn import NeuralNetwork
from utils import load_data, minmax_scale

noise_list = [0.2, 0.3, 0.9]

# ----------- Logistic Regression Classifier -----------
LogReg = LogisticRegression(lr=0.0001, epochs=100)  # instanciate logistic regression class
for i in range(len(noise_list)):  # iterating over datasets with different noise ratios

    print('-------- Logostic Regression on dataset with noise ratio {}--------'.format(noise_list[i]))

    # extract and preprocess train and test data
    x_train, y_train, x_test, y_test = load_data(noise_list[i])
    x_train = minmax_scale(x_train)

    # training process with gradient descent algorithm
    print('#### Training ####')
    loss_list, acc_list = LogReg.grad_desc(x_train, y_train)
    print('{}\n{}'.format(loss_list, acc_list))

    # evaluate model with test dataset
    print('#### Evaluation on test set ####')
    print('weights + bias: {}'.format(LogReg.w))
    x_test = minmax_scale(x_test)
    # compute recall accuracy
    recall_acc = LogReg.predict(x_test, y_test)
    print('Recall Accuracy on test set: {} % \n'.format(recall_acc))

print('-----------------------------------------------------------------------------------------------------------\n')
# ----------- One Hidden Layer Neural Network Classifier -----------
for k in range(len(noise_list)):

    print('-------- One Hidden Layer Neural Network on dataset with noise ratio {}--------'.format(noise_list[k]))

    x_train, y_train, x_test, y_test = load_data(noise_list[k])
    NN = NeuralNetwork(hidden_layer=1, layer_units=[50], lr=0.5, epochs=200)
    NN.init_network(x_train)
    loss_list = ['loss:']
    acc_list = ['acc:']
    print('#### Training ####')
    for j in range(NN.epochs):
        loss, acc = NN.training_step(x_train, y_train)
        loss_list.append(np.round(loss, 3))
        acc_list.append(np.round(acc, 3))
    print('{}\n{}'.format(loss_list, acc_list))
    print('#### Evaluation on test set ####')
    acc = NN.predict(x_test, y_test)
    print('Recall Accuracy on test set: {} % \n'.format(acc))

print('-----------------------------------------------------------------------------------------------------------\n')
# ----------- Two Hidden Layer Neural Network Classifier -----------
for t in range(len(noise_list)):

    print('-------- Two Hidden Layer Neural Network on dataset with noise ratio {}--------'.format(noise_list[t]))

    x_train, y_train, x_test, y_test = load_data(noise_list[t])
    NN = NeuralNetwork(hidden_layer=2, layer_units=[20, 20], lr=0.5, epochs=200)
    NN.init_network(x_train)
    loss_list = ['loss:']
    acc_list = ['acc:']
    print('#### Training ####')
    for l in range(NN.epochs):
        loss, acc = NN.training_step(x_train, y_train)
        loss_list.append(np.round(loss, 3))
        acc_list.append(np.round(acc, 3))
    print('{}\n{}'.format(loss_list, acc_list))
    print('#### Evaluation on test set ####')
    acc = NN.predict(x_test, y_test)
    print('Recall Accuracy on test set: {} % \n'.format(acc))

print('-----------------------------------------------------------------------------------------------------------\n')
# ----------- One Hidden Layer Neural Network Classifier (# of neurons) -----------
neuron_list = [1, 3, 5, 7, 9]
for p in range(len(neuron_list)):
    for g in range(len(noise_list)):

        print('-------- One Hidden Layer Neural Network on dataset with noise ratio {}--------'.format(noise_list[g]))

        x_train, y_train, x_test, y_test = load_data(noise_list[g])
        print('Number of neurons: {}'.format(neuron_list[p]))
        NN = NeuralNetwork(hidden_layer=1, layer_units=[neuron_list[p]], lr=0.5, epochs=200)
        NN.init_network(x_train)
        for j in range(NN.epochs):
            loss, acc = NN.training_step(x_train, y_train)
        acc = NN.predict(x_test, y_test)
        print('Recall Accuracy on test set: {} % \n'.format(acc))
