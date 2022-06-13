# import the necessary packages
from utilities.nn.conv import LeNet
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model #mz
import pdb #mz pdb.set_trace() #mz

# ****** Loading Trained Model ******
model = load_model('out/model0')


# ****** Loading Saved Handwritten Dataset ******
data_5 = np.load('data/data.npy').astype("float32")
data_sc = np.load('data/data_sc.npy').astype("float32")
numLabels = np.load('data/labels.npy').astype("uint8")
numLabels_1d = np.load('data/labels.npy').astype("uint8")

# convert the labels from integers to vectors
le = LabelBinarizer()
numLabels = le.fit_transform(numLabels)


prec_5=np.zeros([3,data_5.shape[0]])
prec_sc=np.zeros([3,data_sc.shape[0]])
res=[prec_5,prec_sc]

for ind, data in enumerate([data_5,data_sc]):
    for k, numData in enumerate(data):
        numData = numData / 255.0
        numData = numData.reshape((numData.shape[0], 28, 28, 1))

    # evaluate the network with nums
        print("[INFO] evaluating network with handwritten numbers...")
        predictions = model.predict(numData, batch_size=32)
        print(classification_report(numLabels.argmax(axis=1),
predictions.argmax(axis=1), target_names= [str(x) for x in le.classes_]))
        if ind==0 and k==4:
            pred=predictions
        rep=classification_report(numLabels.argmax(axis=1),
predictions.argmax(axis=1), target_names= [str(x) for x in le.classes_], output_dict=True)
        res[ind][0,k]=rep['macro avg']['precision']
        res[ind][1,k]=rep['macro avg']['recall']
        res[ind][2,k]=rep['macro avg']['f1-score']


# ****** Demo: Labelling digits ******
import cv2
img = cv2.imread('data/digits.png',1);
delta=img.shape[1]//10
height=np.arange(0,img.shape[0],delta)
width=np.arange(0,img.shape[1],delta)
font = cv2.FONT_HERSHEY_SIMPLEX;

for h_ind, h in enumerate(height[0:15]):
    for w_ind, w in enumerate(width[0:10]):
        label_true=w_ind;
        label_model=pred.argmax(axis=1)[(h_ind)*10+w_ind]
        if label_model==label_true:
            color=(0, 100, 0)
        else:
            color=(0, 0, 100)
        cv2.putText(img,str(label_model),(w,h), font, 4, color, 6, cv2.LINE_AA);

cv2.imwrite('out/pad_15_labeled.png',img);

# ****** Figure 1 ******
prec=res[0]
plt.style.use("ggplot")
plt.figure();
plt.bar(np.arange(0, prec.shape[1])+0.2,prec[0,:],width=0.2, label='precision');
plt.bar(np.arange(0, prec.shape[1])+0.0,prec[1,:],width=0.2, label='recall');
plt.bar(np.arange(0, prec.shape[1])-0.2,prec[2,:],width=0.2, label='f1-score');
plt.legend();
plt.xticks(np.arange(0, prec.shape[1]),['none', 'thresh', 'crop_tight', 'crop_sq', 'pad_1.5'])
plt.title("Trained LeNet on Handwritten Digits \n Testing dataset: 150 handwritten digits x 5 preprocessing styles");
plt.xlabel("Preprocessing Style");
plt.ylabel("Classification Results");
plt.savefig('out/fig1.png');

# ****** Figure 2 ******
prec=res[1]
plt.style.use("ggplot")
plt.figure();
plt.bar(np.arange(0, prec.shape[1]),prec[0,:],width=0.4, label='precision');
#plt.bar(np.arange(0, prec.shape[1])+0.0,prec[1,:],width=0.2, label='recall');
#plt.bar(np.arange(0, prec.shape[1])-0.2,prec[2,:],width=0.2, label='f1-score');
plt.legend();
plt.xticks(np.arange(0, prec.shape[1]),['1.0', '1.1', '1.2', '1.3','1.4','1.5','1.6','1.7','1.8','1.9','2.0'])
plt.title("Trained LeNet on Handwritten Digits \n Testing dataset: 150 handwritten digits x 11 padding scales");
plt.xlabel("Padding Scales");
plt.ylabel("Classification Results");
plt.savefig('out/fig2.png');

#pdb.set_trace() #mz
