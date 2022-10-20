from cgi import test
from gzip import FNAME
from termios import OFDEL
import numpy as np
import h5py
import imageio
import scipy
import matplotlib.pyplot as plt
from PIL import Image


def load_data():
    train_dataset = h5py.File('./datasets/train_catvnoncat.h5', 'r')
    train_set_x_orig = np.array(train_dataset["train_set_x"])
    train_set_y_orig = np.array(train_dataset["train_set_y"])

    test_dataset = h5py.File('./datasets/test_catvnoncat.h5', 'r')
    test_set_x_orig = np.array(test_dataset["test_set_x"])
    test_set_y_orig = np.array(test_dataset["test_set_y"])

    classes = np.array(test_dataset['list_classes'])
    train_set_x_orig = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).transpose()
    test_set_x_orig = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).transpose()
    train_set_x = train_set_x_orig/255.
    test_set_x = test_set_x_orig/255.
    return train_set_x, train_set_y_orig, test_set_x, test_set_y_orig, classes


train_set_x, train_set_y_orig, test_set_x, test_set_y_orig, classes = load_data()
def sigmoid(x):
    z = 1.0 /(1.0 + np.exp(-1.0 * x))
    return z
def initialize_with_zeros(dim):
    w = np.zeros((dim,1))
    b = 0
    return w,b
def propagate(x,y,w,b):
    m = x.shape[1]
    A =sigmoid(np.dot(w.T,x)+b)

    cost = (-1.0 / m) * (np.sum(y*np.log(A)+(1-y)*np.log(1-A)))

    dw = (1.0/m)*np.dot(x,(A-y).T)
    db = (1.0/m)*np.sum(A-y)

    return cost,dw,db

def optimize(w,b,x,y,learning_rate,num_iterations,print_cost=False):
    costs = []
    
    for i in range(num_iterations):
        cost,dw,db = propagate(x,y,w,b)
        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i % 100 == 0:
            costs.append(cost)
        if print_cost & i % 100 == 0:
            print("cost after iterations %i : %f" %(i,cost) )
    return w,b,dw,db,costs

def predict(w,b,x):
    m = x.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(x.shape[0],1)
    A = sigmoid(np.dot(w.T,x)+b)
    for i in range(A.shape[1]):
        if A[0,i]>0.5:
            Y_prediction[0,i] = 1
        else:
            Y_prediction[0,i] = 0
    return Y_prediction
def model(X_train, Y_train,X_test,Y_test,num_iterations,learning_rate,print_cost=False):
    w,b = initialize_with_zeros(X_train.shape[0])
    w,b,dw,db,costs = optimize(w,b,X_train,Y_train,learning_rate,num_iterations,print_cost)
    Y_prediction_test =  predict(w,b,X_test)
    Y_prediction_train =  predict(w,b,X_train)

    print("train accuracy:{}".format(100-np.mean(np.abs(Y_prediction_train-Y_train))*100))
    print("test accuracy:{}".format(100-np.mean(np.abs(Y_prediction_test-Y_test))*100))
    d = {"costs" :costs,"Y_prediction_test":Y_prediction_test,"Y_prediction_train":Y_prediction_train,"w":w,"b":b,"learning_rate":learning_rate,"num_iterations":num_iterations}
    return d

d=model(train_set_x, train_set_y_orig,test_set_x,test_set_y_orig,num_iterations=2000,learning_rate=0.005,print_cost=False)

# index = 0
# plt.imshow(test_set_x[:,index].reshape(64,64,3))
# print("y="+str(test_set_y_orig[index]),"and your predicion is y = \""+classes[int(d["Y_prediction_test"][0,index])].decode("utf-8") + "\" picture.")
# plt.show()
image = np.array(imageio.imread("./myimage.jpeg"))
image = np.array(Image.fromarray(image).resize((64,64)))
plt.imshow(image)
plt.show()
image = image.reshape(1,64*64*3).T
my_predicted_image = predict(d["w"], d["b"], image)


print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")