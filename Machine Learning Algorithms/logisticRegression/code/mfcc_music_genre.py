__author__ = 'Thilak'
import copy
import scipy
from scipy.io import wavfile
from sklearn import cross_validation
import numpy as np
from math import exp
from sklearn.metrics import confusion_matrix
from scikits.talkbox.features import mfcc
import matplotlib.pyplot as plt


#lOADS WAV DATA FROM THE FILE LOCATION AND CONVERT IT TO MFCC COMPONENTS  AND STORE IT AS AN ARRAY AND GENRE ALSO STORED AS AN ARRAY EACH GENRE IS GIVEN A VARIABLE

def loadMFCCData():
    data=[]
    genre_list=[]
    for i in range(0,100):
        if i <10:
           file_path='C:/Users/Thilak/Desktop/ml/pjt 3/pjt3code/opihi.cs.uvic.ca/sound/genres/classical/classical.0000'+str(i)+'.wav'
        else:
           file_path='C:/Users/Thilak/Desktop/ml/pjt 3/pjt3code/opihi.cs.uvic.ca/sound/genres/classical/classical.000'+str(i)+'.wav'
        sample_rate,x=scipy.io.wavfile.read(file_path)
        ceps, mspec, spec = mfcc(x)
        num_ceps = ceps.shape[0]
        mfcc_features = np.mean( ceps[int( num_ceps * 1 / 10 ):int( num_ceps * 9 / 10 )] , axis=0 )
        data.append(mfcc_features)
        genre_list.append(1)

    for i in range(0,100):
        if i <10:
           file_path='C:/Users/Thilak/Desktop/ml/pjt 3/pjt3code/opihi.cs.uvic.ca/sound/genres/jazz/jazz.0000'+str(i)+'.wav'
        else:
           file_path='C:/Users/Thilak/Desktop/ml/pjt 3/pjt3code/opihi.cs.uvic.ca/sound/genres/jazz/jazz.000'+str(i)+'.wav'
        sample_rate,x=scipy.io.wavfile.read(file_path)
        ceps, mspec, spec = mfcc(x)
        num_ceps = ceps.shape[0]
        mfcc_features = np.mean( ceps[int( num_ceps * 1 / 10 ):int( num_ceps * 9 / 10 )] , axis=0 )
        data.append(mfcc_features)
        genre_list.append(2)

    for i in range(0,100):
        if i <10:
           file_path='C:/Users/Thilak/Desktop/ml/pjt 3/pjt3code/opihi.cs.uvic.ca/sound/genres/country/country.0000'+str(i)+'.wav'
        else:
           file_path='C:/Users/Thilak/Desktop/ml/pjt 3/pjt3code/opihi.cs.uvic.ca/sound/genres/country/country.000'+str(i)+'.wav'
        sample_rate,x=scipy.io.wavfile.read(file_path)
        ceps, mspec, spec = mfcc(x)
        num_ceps = ceps.shape[0]
        mfcc_features = np.mean( ceps[int( num_ceps * 1 / 10 ):int( num_ceps * 9 / 10 )] , axis=0 )
        data.append(mfcc_features)
        genre_list.append(3)

    for i in range(0,100):
        if i <10:
           file_path='C:/Users/Thilak/Desktop/ml/pjt 3/pjt3code/opihi.cs.uvic.ca/sound/genres/pop/pop.0000'+str(i)+'.wav'
        else:
           file_path='C:/Users/Thilak/Desktop/ml/pjt 3/pjt3code/opihi.cs.uvic.ca/sound/genres/pop/pop.000'+str(i)+'.wav'
        sample_rate,x=scipy.io.wavfile.read(file_path)
        ceps, mspec, spec = mfcc(x)
        num_ceps = ceps.shape[0]
        mfcc_features = np.mean( ceps[int( num_ceps * 1 / 10 ):int( num_ceps * 9 / 10 )] , axis=0 )
        data.append(mfcc_features)
        genre_list.append(4)

    for i in range(0,100):
        if i <10:
           file_path='C:/Users/Thilak/Desktop/ml/pjt 3/pjt3code/opihi.cs.uvic.ca/sound/genres/rock/rock.0000'+str(i)+'.wav'
        else:
           file_path='C:/Users/Thilak/Desktop/ml/pjt 3/pjt3code/opihi.cs.uvic.ca/sound/genres/rock/rock.000'+str(i)+'.wav'
        sample_rate,x=scipy.io.wavfile.read(file_path)
        ceps, mspec, spec = mfcc(x)
        num_ceps = ceps.shape[0]
        mfcc_features = np.mean( ceps[int( num_ceps * 1 / 10 ):int( num_ceps * 9 / 10 )] , axis=0 )
        data.append(mfcc_features)
        genre_list.append(5)

    for i in range(0,100):
        if i <10:
           file_path='C:/Users/Thilak/Desktop/ml/pjt 3/pjt3code/opihi.cs.uvic.ca/sound/genres/metal/metal.0000'+str(i)+'.wav'
        else:
           file_path='C:/Users/Thilak/Desktop/ml/pjt 3/pjt3code/opihi.cs.uvic.ca/sound/genres/metal/metal.000'+str(i)+'.wav'
        sample_rate,x=scipy.io.wavfile.read(file_path)
        ceps, mspec, spec = mfcc(x)
        num_ceps = ceps.shape[0]
        mfcc_features = np.mean( ceps[int( num_ceps * 1 / 10 ):int( num_ceps * 9 / 10 )] , axis=0 )
        data.append(mfcc_features)
        genre_list.append(6)

    return data,genre_list

# METHOD RETURNS 10 FOLD CROSS VALIDATION DATASET INDICES
def ten_fold_validation(data,ten):
    data_value = data[:]
    fold_data = []
    kf = cross_validation.KFold(600, n_folds=ten,shuffle=True)
    train_data = []
    test_data = []
    for train_index, test_index in kf:
        train_data.append( train_index)
        test_data.append( test_index )
        fold_data.append( (train_data, test_data) )
    return fold_data

# NORMALIZATION OF THE DATASET WITH RESPECT TO MAX VALUE COLUMN WISE(FEATURE WISE)
def normalization(data):
    normalized_data = (data) / data.max(axis=0)
    return normalized_data


#GRADIENT DESCENT WEIGHT UPDATE IN TRAINING DATASET
def training(weight_matrix,new_eta,Delta,norm_train,prob_train,lam_init):
    weight_matrix=weight_matrix+(new_eta*(np.dot((Delta-prob_train),norm_train)-(lam_init*weight_matrix)))
    return weight_matrix


#GETTING THE ACTUAL DATASET FROM THE INDICES RETUREND FROM 10 FOLD CROSS VALIDATION METHOD
def getData(i,train_data_indeces,test_data_indeces,train_data,test_data,Delta,genre_list,updated_data):
    for j in range(len(train_data_indeces[i])):
            train_data[j]=updated_data[train_data_indeces[i][j]]
            Delta[int(genre_list[train_data_indeces[i][j]]-1)][j]=1
    for j in range(len(test_data_indeces[i])):
            test_data[j]=updated_data[test_data_indeces[i][j]]
    return train_data,test_data,Delta

#LOGISTIC REGRESSION IMPLEMENTATION
def cal_Prob_init(data,weight_matrix):
    norm_trans=data.transpose()
    Prob_trans=np.power(np.e, np.dot(weight_matrix, norm_trans))
    Prob_trans[len(Prob_trans)-1]=1
    Prob_trans=Prob_trans/Prob_trans.sum(axis=0)
    #Prob_trans=normalization(Prob_trans)
    return Prob_trans

#MAIN METHOD-START OF THE PROGRAM WITH ETA RATE LAMDA ,WEIGHT INITIALIZATION
def main():
    eta_init = 0.01
    lam_init = 0.001
    iteration_epoch  = 1000
    accuracyList_epoch = []
      # LOAD MFCC DATA FROM THE WAV FILES AND STORE IT IN DATA1 AND THEIR RESPECTIVE GENRES IN GENRE_LIST
    data1,genre_list=loadMFCCData()
    #genre_list=np.load('genre_mfcc.npy')
     #CALLING 10 FOLD CROSS VALIDATION METHOD
    fold_split_data = ten_fold_validation( data1, 10 )
    #print len(fold_split_data)
    example=np.ones((len(data1),1))
    updated_data=np.append(example,data1,1)
    confusion=np.zeros((6,6), dtype=int)
    final_confusion_matrix=np.zeros((6,6), dtype=int)
    #FOR EACH FOLD REPEAT THE SAME STEPS OF TRAINING AND TESTING
    for i in range (len(fold_split_data)):
        weights_init = np.zeros(shape=(6,14))
        train_data_indeces,test_data_indeces  = fold_split_data[i]
        train_data=np.zeros((540,14),dtype=float)
        test_data=np.zeros((60,14),dtype=float)
        Delta=np.zeros((6,540),dtype=int)
        classify=np.zeros((len(test_data_indeces[0]),1), dtype=int)
        #CALLING GET DATA METHOD THAT WILL RETURN ACTUAL DATA FOR TRAINING AND TESTING BASED ON INDICES RETURNED
        train_data,test_data,Delta=getData(i,train_data_indeces,test_data_indeces,train_data,test_data,Delta,genre_list,updated_data)
        #NORMALIZING TRAINING AND TESTING DATA
        norm_train=normalization(train_data)
        norm_test=normalization(test_data)
        weight_matrix=weights_init[:]
        prob_train=cal_Prob_init(norm_train,weight_matrix)
        #print prob_train.shape
        #REPEATING FOR 1000 EPOCHS  TO GET AN UPDATED WEIGHT MATRIX
        for j in range(iteration_epoch):
            new_eta=eta_init/(1+(j/1000))
            weight_matrix=training(weight_matrix,new_eta,Delta,norm_train,prob_train,lam_init)
            prob_train=cal_Prob_init(norm_train,weight_matrix)
         # TEST DATA PROBABILITY MATRIX
        prob_test=cal_Prob_init(norm_test,weight_matrix)
        prob_test=prob_test.transpose()
        k=0
        # CLASSIFY TEST DATA WITH RESPECT TO ITS GENRE VARIABLES
        for k in range(len(test_data_indeces[i])):
            classify[k]=np.argmax(prob_test[k])+1
        j=0
        # CONFUSION MATRIX CALCULATION
        for j in range(len(test_data_indeces[i])):
            confusion[int(genre_list[test_data_indeces[i][j]]-1)][int(classify[j]-1)]=confusion[int(genre_list[test_data_indeces[i][j]]-1)][int(classify[j]-1)]+1
        accuracy=sum(confusion.diagonal())/float(len(norm_test))
        #accuracyList_epoch.append(accuracy)
        #print accuracyList_epoch
        print confusion
    final_confusion_matrix=np.add(final_confusion_matrix,confusion)
    #FINAL CONFUSION MATRIX AND ITS ACCURACY
    print final_confusion_matrix
    #plot_confusion_matrix(final_confusion_matrix)
    final_accuracy=sum(final_confusion_matrix.diagonal())/float(600)
    print final_accuracy


if __name__== '__main__':
    main()