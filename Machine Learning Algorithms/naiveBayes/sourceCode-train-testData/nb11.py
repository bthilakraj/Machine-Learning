#author thilak
import numpy.matlib
import numpy as np
from numpy import matrix
from scipy.sparse import lil_matrix
import matplotlib.pyplot as plt

i = 0
print 'nb'
#method for naive bayes classifier
def nb(beta):
    #initial declaration of class label,vocabulary list,trainig size and esting size
     news20class = 20;
     vocab = 61188;
     trainingsize = 11269;
     testingsize = 7505;
     beta1 = beta#beta value retrieved from the main method where beta=1/|V|
     alpha = 1 + beta1
     #print beta
     #print alpha
    #count declaration for calculating MLE
     count1 = 0;
     count2 = 0;
     count3 = 0;
     count4 = 0;
     count5 = 0;
     count6 = 0;
     count7 = 0;
     count8 = 0;
     count9 = 0;
     count10 = 0;
     count11 = 0;
     count12 = 0;
     count13 = 0;
     count14 = 0;
     count15 = 0;
     count16 = 0;
     count17 = 0;
     count18 = 0;
     count19 = 0;
     count20 = 0;
     
     #MLE CALCULATION IT NEEDS DATA FROM TRAIN.LABEL
     with open('train.label') as f1:
         lines1 = f1.readlines()
     
     for i in lines1:
         #print i
         if i == '1\n':
             count1 = count1 + 1
         elif i == '2\n':
             count2 = count2 + 1
         elif i == '3\n':
             count3 = count3 + 1
         elif i == '4\n':
             count4 = count4 + 1
         elif i == '5\n':
             count5 = count5 + 1
         elif i == '6\n':
             count6 = count6 + 1
         elif i == '7\n':
             count7 = count7 + 1
         elif i == '8\n':
             count8 = count8 + 1
         elif i == '9\n':
             count9 = count9 + 1
         elif i == '10\n':
             count10 = count10 + 1
         elif i == '11\n':
             count11 = count11 + 1
         elif i == '12\n':
             count12 = count12 + 1
         elif i == '13\n':
             count13 = count13 + 1
         elif i == '14\n':
             count14 = count14 + 1
         elif i == '15\n':
             count15 = count15 + 1
         elif i == '16\n':
             count16 = count16 + 1
         elif i == '17\n':
             count17 = count17 + 1
         elif i == '18\n':
             count18 = count18 + 1
         elif i == '19\n':
             count19 = count19 + 1
         elif i == '20\n':
             count20 = count20 + 1
         elif i == '20':
             count20 = count20 + 1
     
     
     #MLE CALCULATION
     prior1 = float(count1) / float(trainingsize);
     prior2 = float(count2) / float(trainingsize);
     prior3 = float(count3) / float(trainingsize);
     prior4 = float(count4) / float(trainingsize);
     prior5 = float(count5) / float(trainingsize);
     prior6 = float(count6) / float(trainingsize);
     prior7 = float(count7) / float(trainingsize);
     prior8 = float(count8) / float(trainingsize);
     prior9 = float(count9) / float(trainingsize);
     prior10 = float(count10) / float(trainingsize);
     prior11 = float(count11) / float(trainingsize);
     prior12 = float(count12) / float(trainingsize);
     prior13 = float(count13) / float(trainingsize);
     prior14 = float(count14) / float(trainingsize);
     prior15 = float(count15) / float(trainingsize);
     prior16 = float(count16) / float(trainingsize);
     prior17 = float(count17) / float(trainingsize);
     prior18 = float(count18) / float(trainingsize);
     prior19 = float(count19) / float(trainingsize);
     prior20 = float(count20) / float(trainingsize);
     
     #MLE IN 20X1 MATRIX
     mle1 = [prior1, prior2, prior3, prior4, prior5, prior6, prior7, prior8, prior9, prior10, prior11, prior12, prior13,
            prior14, prior15, prior16, prior17, prior18, prior19, prior20]
     mle= matrix ([[prior1],[prior2],[prior3],[prior4],[prior5],[prior6],[prior7],[prior8],[prior9],[prior10],[prior11],[prior12],[prior13],[prior14],[prior15],[prior16],[prior17],[prior18],[prior19],[prior20]])
     #NUMPY ARRAY DECLARATION FOR CALCULATION
     sumc = np.zeros(shape=(20,1))
     mapinit = np.zeros(shape=(20,61188))
     map1 = np.zeros(shape=(20,61188))
     confusionmatrix=np.zeros(shape=(20,20))
    #READING DATA FROM INPUT FILES
     traindata1 = np.genfromtxt('train.data', dtype=int, delimiter=' ')
     trainlabel1= np.genfromtxt('train.label', dtype=int, delimiter=' ')
     vocablist1 = np.genfromtxt('vocabulary.txt',dtype=str, delimiter=' ')

     trainlabel1 = trainlabel1[np.newaxis].T
    #MAP CALCULATION USING NUMPY INDICES
     for i in range(0,1467345):
         mapinit[(trainlabel1[(traindata1[i][0])-1][0])-1][(traindata1[i][1])-1] = mapinit[(trainlabel1[(traindata1[i][0])-1][0])-1][(traindata1[i][1])-1] + ((traindata1[i][2]))
     for i in range(0,20):
         sumc[i]= float(sum(mapinit[i])+(alpha-1)*float(vocab))
     #print sumc
     for i in range(0,20):
         for j in range(0,61188):
             map1[i][j]=(float(mapinit[i][j]+(alpha-1))/sumc[i])
     
     #CLASSIFY CALCULATION STEPS
     maplog2=np.log2(map1)

     testdata1=np.genfromtxt('test.data', dtype=int, delimiter=' ')
     testlabel1=np.genfromtxt('test.label', dtype=int, delimiter=' ')
    #SPARSE MATRIX FOR CLASSIFICATION
     testdatamatrix=lil_matrix((7505,61188),dtype=np.float32)
     #print testdatamatrix
     testdata11=testdata1[:,0]-1
     #print testdata11
     testdata12=testdata1[:,1]-1
     #print testdata12
     testdatamatrix[testdata11,testdata12]=testdatamatrix[testdata11,testdata12]+testdata1[:,2]

     testdatamatrix_transpose=testdatamatrix.transpose()#CLASSIFY STEPS ON TEST DATA USING MLE AND MAP
     testdatacalc=maplog2*testdatamatrix_transpose
     testdatacalc=testdatacalc+np.matlib.repmat(mle,1,7505)
     argmaxval=(testdatacalc.argmax(axis=0)+1)
     argmaxval = argmaxval.transpose()
     #CONFUSION MATRIX
     for i in range(0,7505):
         confusionmatrix[(testlabel1[i])-1][(argmaxval[i][0])-1]=confusionmatrix[(testlabel1[i])-1][(argmaxval[i][0])-1]+1
     print confusionmatrix
     accuracy=sum(confusionmatrix.diagonal())/float(7505)
     accuracy=accuracy*100
     #print "Accuracy:", accuracy
     return accuracy

def main():#MAIN METHOD TO METHOD FOR CALCULATING NAIVE BAYES
    vocab = 61188;
    beta = float(1) / float(vocab)
    acc=nb(beta)
    print acc
if __name__== '__main__':
    main()