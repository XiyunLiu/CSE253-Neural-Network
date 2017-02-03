
# coding: utf-8

# In[298]:

get_ipython().magic(u'matplotlib inline')


# In[2]:

import numpy as np
import matplotlib.pyplot as plt


# In[707]:

import os
import struct
from array import array


class MNIST(object):
    def __init__(self, path='./dataset'):
        self.path = path

        self.test_img_fname = 't10k-images-idx3-ubyte'
        self.test_lbl_fname = 't10k-labels-idx1-ubyte'

        self.train_img_fname = 'train-images-idx3-ubyte'
        self.train_lbl_fname = 'train-labels-idx1-ubyte'

        self.test_images = []
        self.test_labels = []

        self.train_images = []
        self.train_labels = []

    def load_testing(self):
        ims, labels = self.load(os.path.join(self.path, self.test_img_fname),
                                os.path.join(self.path, self.test_lbl_fname))
        ims = map(lambda img: [1]+img, ims) # add offset
        self.test_images = ims
        self.test_labels = labels

        return np.array(ims[:2000]), np.array(labels[:2000])

    def load_training(self):
        ims, labels = self.load(os.path.join(self.path, self.train_img_fname),
                                os.path.join(self.path, self.train_lbl_fname))
        ims = map(lambda img: [1]+img, ims) # add offset
        self.train_images = ims
        self.train_labels = labels

        return np.array(ims[:20000]), np.array(labels[:20000])

    @classmethod
    def load(cls, path_img, path_lbl):
        with open(path_lbl, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049,'
                                 'got {}'.format(magic))

            labels = array("B", file.read())

        with open(path_img, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051,'
                                 'got {}'.format(magic))

            image_data = array("B", file.read())

        images = []
        for i in range(size):
            images.append([0] * rows * cols)

        for i in range(size):
            images[i][:] = image_data[i * rows * cols:(i + 1) * rows * cols]

        return images, labels
  

    def showImage(self, imageArray, title = "", xlabel = ""):
        imageArray = imageArray.reshape((28,28))
        fig = plt.figure()
        plotwindow = fig.add_subplot(111)
        plt.xlabel(title)
        
        plt.imshow(imageArray, cmap='gray')
#         plt.show()


# In[708]:

mnist = MNIST()


# In[175]:

trainingImgs, trainingLabels = mnist.load_training()
testImgs, testLabels = mnist.load_testing()


# In[103]:

# Classify 2 vs 3. Get only 2&3 images from the datasest
trainingTwoThreeData = filter(lambda x: x[1] == 2 or x[1] == 3, zip(trainingImgs,trainingLabels))
np.random.shuffle(trainingTwoThreeData)
trainingTwoThreeImgs = np.array(map(lambda x: x[0], trainingTwoThreeData))/255. 
trainingTwoThreeLabels = np.array(map(lambda x: 1 if x[1] == 2 else 0, trainingTwoThreeData))

testTwoThreeData = filter(lambda x: x[1] == 2 or x[1] == 3, zip(testImgs, testLabels))
testTwoThreeImgs = np.array(map(lambda x: x[0], testTwoThreeData))/255.
testTwoThreeLabels = np.array(map(lambda x: 1 if x[1] == 2 else 0, testTwoThreeData))


# In[367]:

# Classify 2 vs 8. Get only 2&8 images from the datasest
trainingTwoEightData = filter(lambda x: x[1] == 2 or x[1] == 8, zip(trainingImgs,trainingLabels))
np.random.shuffle(trainingTwoEightData)
trainingTwoEightImgs = np.array(map(lambda x: x[0], trainingTwoEightData))/255. 
trainingTwoEightLabels = np.array(map(lambda x: 1 if x[1] == 2 else 0, trainingTwoEightData))

testTwoEightData = filter(lambda x: x[1] == 2 or x[1] == 8, zip(testImgs, testLabels))
testTwoEightImgs = np.array(map(lambda x: x[0], testTwoEightData))/255.
testTwoEightLabels = np.array(map(lambda x: 1 if x[1] == 2 else 0, testTwoEightData))


# In[64]:

def plotUpdate(first, second, third, title, firstLabel, secondLabel, thirdLabel, ylabel, ylim, loc = False):
    firstLine, = plt.plot([i for i in range(0,len(first))], first)
    secontLine, = plt.plot([i for i in range(0,len(second))], second)
    thirdLine, = plt.plot([i for i in range(0,len(third))], third)
    plt.xlabel('t')
    plt.ylim(ylim)
    plt.ylabel(ylabel)
    plt.title(title)
    if loc:
        plt.legend([firstLine, secontLine, thirdLine], [firstLabel, secondLabel, thirdLabel], loc = (0.624,0))
    else:
        plt.legend([firstLine, secontLine, thirdLine], [firstLabel, secondLabel, thirdLabel])
    plt.grid(True)


# In[477]:

class LogisticRegression():
    
    def __init__(self):
        self.stepSize = -1
        self.numFeature = -1
        self.maxIter = 400
        self.coeff = None
        self.lossPathTrain = []
        self.lossPathValidation = []
        self.lossPathTest = []
        self.errorPathTrain = []
        self.errorPathValidation = []
        self.errorPathTest = []
        self.coeffPath = []
        self.coeffLengthPath = []
        
    def fit(self, trainingImages, trainingLabels, testImages = None, testLabels = None, stepSize = 0.001, T = 2000, regularization = 'l1', strength = 0, earlyStop = False):
        numValidation = len(trainingLabels)/10
        
        validationImages = trainingImages[:numValidation]
        validationLabels = trainingLabels[:numValidation]
        
        trainingImages = trainingImages[numValidation:]
        trainingLabels = trainingLabels[numValidation:]
        
        self.numFeature = len(trainingImages[0])
        
        self.coeff = np.ones(self.numFeature) #785
        self.stepSize = stepSize
        self.T = T
        self.train(trainingImages, trainingLabels, validationImages, validationLabels, testImages, testLabels, stepSize, T, self.coeff, regularization, strength, earlyStop)
        self.test(validationImages, validationLabels, self.coeff)
        
    def calculateDerivative(self, X, y, coeff, regularization, strength):
        '''
        X: trainingDatas
        y: trainingLabels
        '''
        derivative = np.zeros(len(X[0]))
        for i in range(0, len(y)):
            derivative += (1. / (1 + np.exp(- np.dot(X[i], coeff))) - y[i]) * X[i]
        if regularization == 'l1':
            l1Derivative = coeff.copy()
            l1Derivative[l1Derivative < 0] = -1
            l1Derivative[l1Derivative > 0] = 1
            derivative += strength * l1Derivative
        elif regularization == 'l2':
            derivative += strength * coeff
        return derivative
    
    def calculateLoss(self, X, y, coeff, regularization, strength):
        loss = 0
        for i in xrange(0, len(y)):
            if y[i] == 1:
                loss -= np.log(1. / (1+ np.exp(- np.dot(X[i], coeff))))
            else:
                loss -= np.log(1. / (1+ np.exp(np.dot(X[i], coeff))))
        loss = loss/len(y)
        if regularization == 'l1':
            loss += strength * np.linalg.norm(coeff, ord = 1)
        elif regularization == 'l2':
            loss += strength * np.dot(coeff, coeff)
        else:
            print "error regularization"
        return loss
    
    def train(self, X, y, validationImages, validationLabels, testImages, testLabels, iniStepSize, T, coeff, regularization, strength, earlyStop):
        numIter = 0
        while numIter <= self.maxIter:
            stepSize = iniStepSize /(1 + numIter / T)
            derivative = self.calculateDerivative(X, y, coeff, regularization, strength)
            coeff -= stepSize * derivative
            numIter += 1
            self.lossPathTrain.append(self.calculateLoss(X, y, coeff, regularization, strength))
            self.lossPathValidation.append(self.calculateLoss(validationImages, validationLabels, coeff, regularization, strength))
            self.lossPathTest.append(self.calculateLoss(testImages, testLabels, coeff, regularization, strength))
            
            self.errorPathTrain.append(self.test(X, y, coeff))
            self.errorPathValidation.append(self.test(validationImages, validationLabels, coeff))
            self.errorPathTest.append(self.test(testImages, testLabels, coeff))
            
            self.coeffPath.append(coeff[:])
            self.coeffLengthPath.append(np.dot(coeff,coeff))
            
            if earlyStop and len(self.errorPathValidation) > 40 and              self.errorPathValidation[-1] < self.errorPathValidation[-2] < self.errorPathValidation[-3] :
                break;
                
        self.coeff = coeff
        
        if earlyStop:
            self.coeff = self.coeffPath[-3]
        return self.coeff
            
    def test(self, X, y, coeff):
        '''
        return correction rate
        '''
        error = 0
        for i in range(0, len(y)):
            prob = 1. / (1+ np.exp(- np.dot(X[i], coeff)))
            if (prob >=0.5 and y[i] == 0) or (prob < 0.5 and y[i] == 1):
                error += 1
        return 1-error*1.0/len(y)                


# In[355]:

# 2 vs 3
logisticRegression = LogisticRegression()
logisticRegression.fit(trainingTwoThreeImgs, trainingTwoThreeLabels, testTwoThreeImgs, testTwoThreeLabels,                       stepSize = 0.005, T = 500, earlyStop = True)


# In[356]:

# plt.figure(figsize=(15,4))
# plt.subplot(1,2,1)
plotUpdate(logisticRegression.lossPathTrain, logisticRegression.lossPathValidation, 
           logisticRegression.lossPathTest, 'Loss function (E) over training',
           'training set', 'validation set','test set', "Loss", [0,5])
plt.savefig('Logistiv_2_vs_3_loss.png', bbox_inches='tight')
# plt.subplot(1,2,2)
# plotUpdate(logisticRegression.errorPathTrain, logisticRegression.errorPathValidation, 
#            logisticRegression.errorPathTest, 'Correction rate over training',
#            'training set', 'validation set','test set', "percent correct classification", [0.8, 1], True)
# plt.savefig('Logistiv_2_vs_3_rate.png', bbox_inches='tight')


# In[357]:

plotUpdate(logisticRegression.errorPathTrain, logisticRegression.errorPathValidation, 
           logisticRegression.errorPathTest, 'Correction rate over training',
           'training set', 'validation set','test set', "percent correct classification", [0.8, 1], True)
plt.savefig('Logistiv_2_vs_3_rate.png', bbox_inches='tight')


# In[720]:

mnist.showImage(logisticRegression.coeff[1:])
plt.savefig('TwoThreeWeight.png', bbox_inches='tight')


# In[723]:

logisticRegression.errorPathTest[-3]


# In[411]:

# 2 vs 8
logisticRegression2 = LogisticRegression()
logisticRegression2.fit(trainingTwoEightImgs, trainingTwoEightLabels, testTwoEightImgs, testTwoEightLabels,
                        stepSize = 0.002, T = 100, earlyStop = True)


# In[415]:

# plt.figure(figsize=(15,4))
# plt.subplot(1,2,1)
plotUpdate(logisticRegression2.lossPathTrain, logisticRegression2.lossPathValidation, 
           logisticRegression2.lossPathTest, 'Loss function (E) over training',
           'training set', 'validation set','test set', "Loss", [0,5])
plt.savefig('Logistiv_2_vs_8_loss.png', bbox_inches='tight')
# plt.subplot(1,2,2)
# plotUpdate(logisticRegression2.errorPathTrain, logisticRegression2.errorPathValidation, 
#            logisticRegression2.errorPathTest, 'Correction rate over training',
#            'training set', 'validation set','test set', "percent correct classification", [0.8, 1], True)


# In[414]:

# plt.subplot(1,2,2)
plotUpdate(logisticRegression2.errorPathTrain, logisticRegression2.errorPathValidation, 
           logisticRegression2.errorPathTest, 'Correction rate over training',
           'training set', 'validation set','test set', "percent correct classification", [0.8, 1], True)
plt.savefig('Logistiv_2_vs_8_rate.png', bbox_inches='tight')


# In[721]:

mnist.showImage(logisticRegression2.coeff[1:])
plt.savefig('TwoEightWeight.png', bbox_inches='tight')


# In[722]:

logisticRegression2.errorPathTest[-3]


# In[615]:

# L2 Regularization
logisticRegression_L2 = LogisticRegression()
logisticRegression_L2.fit(trainingTwoThreeImgs, trainingTwoThreeLabels, testTwoThreeImgs, testTwoThreeLabels,                       stepSize = 0.005, T = 100, earlyStop = True, regularization = 'l2', strength = 4)


# In[594]:

logisticRegression_L2_2 = LogisticRegression()
logisticRegression_L2_2.fit(trainingTwoThreeImgs, trainingTwoThreeLabels, testTwoThreeImgs, testTwoThreeLabels,                       stepSize = 0.005, T = 100, earlyStop = True, regularization = 'l2', strength = 1)


# In[595]:

logisticRegression_L2_3 = LogisticRegression()
logisticRegression_L2_3.fit(trainingTwoThreeImgs, trainingTwoThreeLabels, testTwoThreeImgs, testTwoThreeLabels,                       stepSize = 0.005, T = 100, earlyStop = True, regularization = 'l2', strength = 0.1)


# In[606]:

logisticRegression_L2_4 = LogisticRegression()
logisticRegression_L2_4.fit(trainingTwoThreeImgs, trainingTwoThreeLabels, testTwoThreeImgs, testTwoThreeLabels,                       stepSize = 0.005, T = 100, earlyStop = True, regularization = 'l2', strength = 0.01)


# In[616]:

logisticRegression_L2_5 = LogisticRegression()
logisticRegression_L2_5.fit(trainingTwoThreeImgs, trainingTwoThreeLabels, testTwoThreeImgs, testTwoThreeLabels,                       stepSize = 0.005, T = 100, earlyStop = True, regularization = 'l2', strength = 0.0001)


# In[651]:

def plotUpdateForFive(first, second, third, forth, fifth, title, firstLabel, secondLabel, thirdLabel, forthLabel, fifthLabel, ylabel, ylim, loc):
    firstLine, = plt.plot([i for i in range(0,len(first))], first)
    secontLine, = plt.plot([i for i in range(0,len(second))], second)
    thirdLine, = plt.plot([i for i in range(0,len(third))], third)
    forthLine, = plt.plot([i for i in range(0,len(forth))], forth)
    fifthLine, = plt.plot([i for i in range(0,len(fifth))], fifth)
    plt.xlabel('t')
    plt.ylim(ylim)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend([firstLine, secontLine, thirdLine, forthLine, fifthLine], [firstLabel, secondLabel, thirdLabel, forthLabel, fifthLabel], loc = loc)
    plt.grid(True)


# In[621]:

# Compare different lambda in L2
plotUpdateForFive(logisticRegression_L2.errorPathTrain, logisticRegression_L2_2.errorPathTrain,
           logisticRegression_L2_3.errorPathTrain,
                  logisticRegression_L2_4.errorPathTrain, logisticRegression_L2_5.errorPathTrain,
                  'Correction rate over training',
            'lambda = 4','lambda = 1','lambda = 0.1', 'lambda = 0.01','lambda = 0.0001', "percent correct classification", [0.8, 1])
plt.savefig('L2_train.png', bbox_inches='tight')


# In[628]:

# Compare different lambda in L2
plotUpdateForFive(logisticRegression_L2.coeffLengthPath, logisticRegression_L2_2.coeffLengthPath, 
           logisticRegression_L2_3.coeffLengthPath, \
                  logisticRegression_L2_4.coeffLengthPath, logisticRegression_L2_5.coeffLengthPath,\
                  'Length of weight vector over training',
            'lambda = 4','lambda = 1','lambda = 0.1', 'lambda = 0.01','lambda = 0.0001', "Length of weight vector", [300, 3400])
plt.savefig('L2_weight_length.png', bbox_inches='tight')


# In[609]:

# L2 regularization
print np.dot(logisticRegression_L2.coeff,logisticRegression_L2.coeff) # lambda = 0.01
print np.dot(logisticRegression_L2_2.coeff,logisticRegression_L2_2.coeff) # lambda = 0.001
print np.dot(logisticRegression_L2_3.coeff,logisticRegression_L2_3.coeff) # lambda = 0.0001
print np.dot(logisticRegression_L2_4.coeff,logisticRegression_L2_4.coeff) # lambda = 0
print np.dot(logisticRegression_L2_5.coeff,logisticRegression_L2_5.coeff) # lambda = 0


# In[629]:

plt.plot(np.log10(np.array([2,1,0.1,0.01,0.001])),1-np.array([ logisticRegression_L2.errorPathTest[-3],
                                                  logisticRegression_L2_2.errorPathTest[-3], 
                                                  logisticRegression_L2_3.errorPathTest[-3],
                                                  logisticRegression_L2_4.errorPathTest[-3],
                                                  logisticRegression_L2_5.errorPathTest[-3]]))
plt.ylabel("test set error")
plt.xlabel("lambda")
plt.savefig('L2_error_lambda.png', bbox_inches='tight')


# In[639]:

# L1 Regularization
# lambda = 1
logisticRegression_L1 = LogisticRegression()
logisticRegression_L1.fit(trainingTwoThreeImgs, trainingTwoThreeLabels, testTwoThreeImgs, testTwoThreeLabels,                       stepSize = 0.005, T = 100, earlyStop = True, regularization = 'l1', strength = 2)


# In[640]:

logisticRegression_L1_2 = LogisticRegression()
logisticRegression_L1_2.fit(trainingTwoThreeImgs, trainingTwoThreeLabels, testTwoThreeImgs, testTwoThreeLabels,                       stepSize = 0.005, T = 100, earlyStop = True, regularization = 'l1', strength = 1)


# In[641]:

logisticRegression_L1_3 = LogisticRegression()
logisticRegression_L1_3.fit(trainingTwoThreeImgs, trainingTwoThreeLabels, testTwoThreeImgs, testTwoThreeLabels,                       stepSize = 0.005, T = 100, earlyStop = True, regularization = 'l1', strength = 0.1)


# In[648]:

logisticRegression_L1_4 = LogisticRegression()
logisticRegression_L1_4.fit(trainingTwoThreeImgs, trainingTwoThreeLabels, testTwoThreeImgs, testTwoThreeLabels,                       stepSize = 0.005, T = 100, earlyStop = True, regularization = 'l1', strength = 0.001)


# In[681]:

logisticRegression_L1_5 = LogisticRegression()
logisticRegression_L1_5.fit(trainingTwoThreeImgs, trainingTwoThreeLabels, testTwoThreeImgs, testTwoThreeLabels,                       stepSize = 0.005, T = 100, earlyStop = True, regularization = 'l1', strength = 0.00001)


# In[682]:

# Compare different lambda in L1
plotUpdateForFive(logisticRegression_L1.errorPathTrain, logisticRegression_L1_2.errorPathTrain,
           logisticRegression_L1_3.errorPathTrain,
                  logisticRegression_L1_4.errorPathTrain, logisticRegression_L1_5.errorPathTrain,
                  'Correction rate over training',
            'lambda = 2','lambda = 1','lambda = 0.1', 'lambda = 0.001','lambda = 0.00001', "percent correct classification", [0.8, 1], loc = [0.54,0.02])
plt.savefig('L1_train.png', bbox_inches='tight')


# In[683]:

# Compare different lambda in L1
plotUpdateForFive(logisticRegression_L1.coeffLengthPath, logisticRegression_L1_2.coeffLengthPath, 
           logisticRegression_L1_3.coeffLengthPath, \
                  logisticRegression_L1_4.coeffLengthPath, logisticRegression_L1_5.coeffLengthPath,\
                  'Length of weight vector over training',
            'lambda = 2','lambda = 1','lambda = 0.1', 'lambda = 0.001','lambda = 0.00001', "Length of weight vector", [800, 3200], loc = [0.536,0.53])
plt.savefig('L1_weight_length.png', bbox_inches='tight')


# In[684]:

# L1 regularization
print np.dot(logisticRegression_L1.coeff, logisticRegression_L1.coeff) # lambda = 0.01
print np.dot(logisticRegression_L1_2.coeff,logisticRegression_L1_2.coeff) # lambda = 0.001
print np.dot(logisticRegression_L1_3.coeff,logisticRegression_L1_3.coeff) # lambda = 0.0001
print np.dot(logisticRegression_L1_4.coeff,logisticRegression_L1_4.coeff) # lambda = 0.0001
print np.dot(logisticRegression_L1_5.coeff,logisticRegression_L1_5.coeff) # lambda = 0.0001


# In[685]:

plt.plot(np.log10(np.array([2,1,0.1,0.001,0.00001])),1-np.array([ logisticRegression_L1.errorPathTest[-3],
                                                  logisticRegression_L1_2.errorPathTest[-3], 
                                                  logisticRegression_L1_3.errorPathTest[-3],
                                                  logisticRegression_L1_4.errorPathTest[-3],
                                                  logisticRegression_L1_5.errorPathTest[-3]]))
plt.ylabel("test set error")
plt.xlabel("lambda")
plt.savefig('L1_error_lambda.png', bbox_inches='tight')


# In[710]:

mnist.showImage(logisticRegression_L1.coeff[1:], title = "(a) Lambda = 2")
plt.savefig('L1_2.png', bbox_inches='tight')


# In[711]:

mnist.showImage(logisticRegression_L1_2.coeff[1:], title = "(b) Lambda = 1")
plt.savefig('L1_1.png', bbox_inches='tight')


# In[712]:

mnist.showImage(logisticRegression_L1_3.coeff[1:], title = "(c) Lambda = 0.1")
plt.savefig('L1_0.1.png', bbox_inches='tight')


# In[713]:

mnist.showImage(logisticRegression_L1_4.coeff[1:] , title = "(d) Lambda = 0.001")
plt.savefig('L1_0.001.png', bbox_inches='tight')


# In[714]:

mnist.showImage(logisticRegression_L1_5.coeff[1:], title = "(e) Lambda = 0.00001")
plt.savefig('L1_0.00001.png', bbox_inches='tight')


# In[715]:

mnist.showImage(logisticRegression_L2.coeff[1:], title = "(a) Lambda = 4")
plt.savefig('L2_4.png', bbox_inches='tight')


# In[716]:

mnist.showImage(logisticRegression_L2_2.coeff[1:], title = "(b) Lambda = 1")
plt.savefig('L2_1.png', bbox_inches='tight')


# In[717]:

mnist.showImage(logisticRegression_L2_3.coeff[1:], title = "(c) Lambda = 0.1")
plt.savefig('L2_0.1.png', bbox_inches='tight')


# In[718]:

mnist.showImage(logisticRegression_L2_4.coeff[1:], title = "(d) Lambda = 0.01")
plt.savefig('L2_0.01.png', bbox_inches='tight')


# In[719]:

mnist.showImage(logisticRegression_L2_5.coeff[1:],title = "(e) Lambda = 0.0001")
plt.savefig('L2_0.0001.png', bbox_inches='tight')


# In[326]:

b = a.copy()


# In[327]:

b


# In[328]:

b[0] = 23


# In[329]:

b


# In[330]:

a


# In[ ]:



