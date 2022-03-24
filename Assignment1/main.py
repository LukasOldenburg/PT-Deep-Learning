from logistic_regression import LogisticRegression
from utils import load_data, minmax_scale

# ----------- Logistic Regression Classifier -----------
LogReg = LogisticRegression(lr=0.0001, epochs=100)  # instanciate logistic regression class
noise_list = [0.2, 0.3, 0.9]
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

# ----------- One Hidden Layer Neural Network Classifier -----------
