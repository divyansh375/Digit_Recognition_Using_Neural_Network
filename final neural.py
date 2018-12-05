import numpy as np
import cv2
from sklearn.neural_network import MLPClassifier
x11=[]

name_of_file=input("enter the name of image file. (Note : the image should be 8x8px)")

n=cv2.imread(name_of_file)
n= cv2.cvtColor(n, cv2.COLOR_BGR2GRAY)

for ii in range(0,8,1):
    x22=[]
    for jj in range(0,8,1):
        
        x22.append(float(n[ii][jj]))
    x11.append(x22)
        




from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_digits
digits = load_digits()
x=digits.images[0:1400]
y=digits.target[0:1400]




nsamples, nx, ny = x.shape
x = x.reshape((nsamples,nx*ny))



clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(20, 10), random_state=1)



clf.fit(x, y)

print(clf)


x=digits.images[1420:1430]
x[0]=x11
nsamples, nx, ny = x.shape
x = x.reshape((nsamples,nx*ny))

import matplotlib.pyplot as plt 
plt.gray() 
plt.matshow(x11) 
plt.show()


new=clf.predict(x)



z="{}".format(new)




print("the digit you have drawn is ")
print(z[1])



