from logistic_regression import LogisticRegression
from utils import load_data, minmax_scale

# evaluation of dataset two_moon_0.2
x_train, y_train, x_test, y_test = load_data(0.2)
LogReg = LogisticRegression()
x_train = minmax_scale(x_train)
x_test = minmax_scale(x_test)
loss_list, acc_list = LogReg.grad_desc(x_train, y_train)
print('Logostic Regression on dataset with noise ratio 0.2 \n{}'.format(loss_list))
print('{} \n'.format(acc_list))

# evaluation of dataset two_moon_0.3
x_train, y_train, x_test, y_test = load_data(0.3)
LogReg = LogisticRegression()
x_train = minmax_scale(x_train)
x_test = minmax_scale(x_test)
loss_list, acc_list = LogReg.grad_desc(x_train, y_train)
print('Logostic Regression on dataset with noise ratio 0.3 \n{}'.format(loss_list))
print('{} \n'.format(acc_list))

# evaluation of dataset two_moon_0.9
x_train, y_train, x_test, y_test = load_data(0.9)
LogReg = LogisticRegression()
x_train = minmax_scale(x_train)
x_test = minmax_scale(x_test)
loss_list, acc_list = LogReg.grad_desc(x_train, y_train)
print('Logostic Regression on dataset with noise ratio 0.9 \n{}'.format(loss_list))
print(acc_list)