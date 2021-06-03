import cv2
import numpy

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml

from PIL import Image
import PIL.ImageOps

x = numpy.load('image.npz')['arr_0']
y = pd.read_csv('labels.csv')['labels']

classes=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
nclasses=len(classes)

samples_per_class = 5
figure=plt.figure(figsize=(nclasses*3,(1+samples_per_class*2)))
index_class = 0

for i in classes:
  index = numpy.flatnonzero(y == i)
  index = numpy.random.choice(index,samples_per_class,replace=False)
  j = 0
  for k in index:
    plot_index = j*nclasses + index_class + 1
    p = plt.subplot(samples_per_class,nclasses,plot_index)  
    p = sns.heatmap(numpy.reshape(x[k],(22,30)),cmap=plt.cm.gray,xticklabels=True,yticklabels=True,cbar=False)
    p = plt.axis('off')
    j = j + 1  
  index_class = index_class + 1

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=7500,test_size=2500,random_state=9)

x_train = x_train/255
x_test = x_test/255

clf=LogisticRegression(solver='saga',multi_class='multinomial').fit(x_train,y_train)

prediction = clf.predict(x_test)
accuracy_score = accuracy_score(y_test, prediction)
accuracy = accuracy_score * 100

print("The accuracy of the prediction is ", accuracy, "%. ")

cm=pd.crosstab(y_test,prediction,rownames=['actual'],colnames=['Predicted'])
p=plt.figure(figsize=(10,10))
p=sns.heatmap(cm,annot=True,fmt='d',cbar=True)

capture = cv2.VideoCapture(0)

while(True):
    try:
        ret, frame = capture.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        upper_left = (int(width/2 - 56), int(width/2 - 56))
        lower_right = (int(width/2 + 56), int(width/2 + 56))
        cv2.rectangle(gray, upper_left, lower_right, (0, 0, 0), 2)

        roi = gray[upper_left[1]: lower_right[1], upper_left[0]: lower_right[0]]
            
        image_pil = Image.fromarray(roi)
        image_l = image_pil.convert("L")
        image_resized = image_l.resize((28, 28), Image.ANTIALIAS())
        image_flip = PIL.ImageOps.invert(image_resized)
            
        pixel_filter = 20
        min_pixel = numpy.percentile(image_flip, pixel_filter)
        image_scale = numpy.clip(image_flip - min_pixel, 0, 255)
        max_pixel = numpy.max(image_flip)
        image_scaled = numpy.asarray(image_scale)/max_pixel

        test_sample = numpy.array(image_scaled).reshape(1, 784)
        test_predict = clf.predict(test_sample)

        print("The predicted class is ", test_predict)
        cv2.imshow("frame", gray)

        capture.release()
        cv2.destroyAllWindows()
    except Exception as e:
        pass