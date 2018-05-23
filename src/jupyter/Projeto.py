# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 22:13:07 2018

@author: G_N17
"""
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
from sklearn.feature_selection import RFE
from sklearn.svm import LinearSVC
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

class Classify(object):
    """docstring for ClassName"""
    def __init__(self, arg):
        self.dataset = self.dataset        

    def data_train(self):
    	file=open("yamanishi_DTIs_REAL_NEGS.txt",'r')
    	file_final=file.readlines()
    	for i in range(len(file_final)):
        	file_final[i]=file_final[i].split()
        	file_final[i].remove(file_final[i][0])
        	file_final[i].remove(file_final[i][0])
    	matrix=np.array(file_final,dtype='float64')
    	labels=matrix[:,len(matrix[0,:])-1].astype('int32')
    	matrix=matrix[:,0:len(matrix[0,:])-1]
    	np.save('matrix_train',matrix)
    	np.save('labels',labels)


    # Kruskal-Wallis
    def Kruskal_Wallis(self, data,labels):
        data_H = []
        data_Pvalues = []
        nr_features=len(data[0,:])

        for i in range(nr_features):
            matrix_feature = data[:,i]
            kw_result = stats.kruskal(matrix_feature, labels)
            data_H.append(kw_result[0])
            data_Pvalues.append(kw_result[1])

        data_Pvalues = np.array(data_Pvalues,dtype='float64')
        data_H = np.array(data_H)
        np.save(path + 'data_H', data_H.astype('float64'))
        np.save(path + 'data_Pvalues', data_Pvalues.astype('float64'))

    def plot_kwallis(self, data_H):
    	figure=plt.figure()
    	plt.plot(-np.sort(-data_H))
    	plt.title('K Wallis Method')
    	plt.xlabel('Features')
    	plt.ylabel('H Values')
    	plt.show()
    	plt.savefig('K_Wallis_Figure')

    def plot_rf(self, features_importance):
    	figure=plt.figure()
    	plt.plot(np.sort(features_importance))
    	plt.title('RF Method')
    	plt.xlabel('Features')
    	plt.ylabel('Features Importance')
    	plt.show()
    	plt.savefig('rf_Figure')

    def plot_rfe(self, rf_rank):
    	figure=plt.figure()
    	plt.plot(np.sort(rf_rank))
    	plt.title('RFE Method')
    	plt.xlabel('Features')
    	plt.ylabel('Features Rank')
    	plt.show()
    	plt.savefig('rfe_Figure')


    def features_selection(self, data_rfe,data_rf,data_kw,number_features):
        scaler = MinMaxScaler()
        r2=abs(scaler.fit_transform(data_rfe[:,None])-1)
        rf2=scaler.fit_transform(data_rf[:,None])
        kw2=scaler.fit_transform(data_kw[:,None])
        features_final=r2+rf2+kw2
        features_final=features_final[:,0]
        
        plt.plot(-np.sort(-features_final))
        plt.title("Features Importance")
        plt.xlabel("Features")
        plt.ylabel("Feature Importance")
        plt.show()
        
        data_sorted=-np.sort(-features_final)
        list_data=list(data_sorted[0:number_features])
        list_position=[]
        for i in range(len(list_data)):
            list_position.append(np.asscalar(np.where(features_final == list_data[i])[0][0]))
            
        return list_position


    def RFE_LinearSVC(self, data,labels,step_value):
    	classifier=LinearSVC()
    	rfe_model=RFE(classifier,step=step_value,n_features_to_select=1)
    	rfe_model=rfe_model.fit(data,labels)
    	pickle.dump( rfe_model, open( "rfe_model.py", "wb" ) )	

    def rf_classifier(self, data,labels,n):
    	clf = RandomForestClassifier(n_estimators=n)
    	clf.fit(data,labels)
    	labels_pred=clf.predict(data)
    	feature_scores=clf.feature_importances_
    	list_info=np.array(feature_scores)
    	print(clf.score)
    	np.save('RF_features_importance',list_info)

    def normal_matrix(self, data):
        matrix_normal=preprocessing.scale(data,axis=0)
        np.save('matrix_normal',matrix_normal.astype('float64'))
       
        
    def mean_std(self, data,option):
        if option=='mean':
            mean=np.mean(data,axis=0)
            np.save('matrix_mean',mean)
        elif option=='std':
            std=np.std(data,axis=0)
            np.save('matrix_std',std)


    def LinearSVM(self, data,labels,cross):
        linearsvc=LinearSVC()
        if cross==False:
            linearsvc.fit(data,labels)
            pickle.dump(linearsvc, open( 'linearsvc.py', 'wb' ) )
        elif cross==True:
            s_kfold=StratifiedKFold(n_splits=5,shuffle=True)
            print(cross_val_score(linearsvc,data,labels,cv=s_kfold))
            

    def rf_classifier(self, data,labels,n,cross):
        randomforest=RandomForestClassifier(n_estimators=n)
        if cross==False:
            randomforest.fit(data,labels)
            pickle.dump(randomforest, open( 'randomforest.py', 'wb' ) )
        elif cross==True:
            s_kfold=StratifiedKFold(n_splits=5,shuffle=True)
            print(cross_val_score(randomforest,data,labels,cv=s_kfold))
            

    def SVC_nonlinear(self, data,labels,kernel,cross,c_value,poly_degree):
        svc=svm.SVC(C=c_value,degree=poly_degree,kernel=kernel,decision_function_shape='ovr')
        if cross==False:
            svc.fit(data,labels)
            pickle.dump(svc, open( 'svc_'+kernel+'_'+'c_value_'+str(c_value)+'_'+'poly_degree_'+str(poly_degree)+'.py', 'wb' ) )
        elif cross==True:
            s_kfold=StratifiedKFold(n_splits=5,shuffle=True)
            print(cross_val_score(svc,data,labels,cv=s_kfold))
        
        

    def KNN(self, data,labels,num_neigh,cross):
        knn=KNeighborsClassifier(n_neighbors=num_neigh)
        if cross==False:
            knn.fit(data,labels)
            pickle.dump(knn, open( 'knn_'+str(num_neigh)+'.py', 'wb' ) )
        elif cross==True:
            s_kfold=StratifiedKFold(n_splits=5,shuffle=True)
            print(cross_val_score(knn,data,labels,cv=s_kfold))

    def accuracy(self, y_label,y_pred):
    	accuracy = accuracy_score(y_label, y_pred)

    def confusion_matrix(self, y_label,y_pred):
    	c_m=confusion_matrix(y_label, y_pred)
    	print(c_m)
    	tn, fp, fn, tp = confusion_matrix(y_label,y_pred)
    	sensitivity=tp/(tp+fn)
    	specificity=tn/(tn+fp)

    def ROC(self, y_label,y_pred,pos):
    	fpr, tpr, thresholds = roc_curve(y_label,y_pred,pos_label=pos)
    	auc = auc(fpr, tpr)
    	print(auc)

