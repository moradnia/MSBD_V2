import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_curve, auc, precision_score,recall_score,f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import time

start = time.time()
st.title('MSBD Uusing ML algorithm')
st.write("""
# Explore different classifier
# Develop By : Milad Moradnia
""")
dataset_name = st.sidebar.selectbox('select dataset',('Twitter','twitter2'))
st.write('dataset:  '+dataset_name)
classifier_name = st.sidebar.selectbox('select classifier',('svm','decition tree','random forest','naive bayes','logistic regression','AdaBoost'))


def get_dataset(dataset_name,path):
    if dataset_name == 'Twitter':
        data = pd.read_csv(path)
    if dataset_name=='twitter2':
        data=pd.read_csv(path)

    X = data.iloc[:,:-1]
    y = data.iloc[:,-1]
    return X,y

# path = 'Twitter_dataset.csv'
path = dataset_name +'.csv'
X,y = get_dataset(dataset_name,path)
st.write('shape of dataset',X.shape)
st.write('number of classess', len(pd.unique(y)))

def add_parameter_ui(clf_name):
    params = {}
    if clf_name == 'svm':
        C = st.sidebar.slider('C', 0.01,1.0)
        params['C'] = C
    elif clf_name == 'decition tree':
        criterion = st.sidebar.selectbox('criterion',('entropy','gini'))
        params['criterion'] = criterion
        min_samples_leaf = st.sidebar.slider('min_samples_leaf',1,55)
        params['min_samples_leaf'] = min_samples_leaf
        min_samples_split = st.sidebar.slider('min_samples_split', 2,12)
        params['min_samples_split'] = min_samples_split
    elif clf_name == 'random forest' :
        criterion = st.sidebar.selectbox('criterion',('entropy','gini'))
        params['criterion'] = criterion
        min_samples_leaf = st.sidebar.slider('min_samples_leaf',1,100)
        params['min_samples_leaf'] = min_samples_leaf
        min_samples_split = st.sidebar.slider('min_samples_split', 2,25)
        params['min_samples_split'] = min_samples_split
    elif clf_name == 'naive bayes' :
        alpha = st.sidebar.slider('alpha',0.0009,1.0)
        params['alpha'] = alpha
    elif clf_name == 'logistic regression':
        penalty = st.sidebar.selectbox('penalty',('l2',))
        params['penalty']=penalty
    elif clf_name == 'AdaBoost':
        n_estimators = st.sidebar.slider('n_estimators',50,130)
        params['n_estimators'] = n_estimators
    return params

params = add_parameter_ui(classifier_name)

def get_classifier(clf_name, params):


    if clf_name == 'svm':
        clf = SVC(random_state=3432,C=params['C'])

    elif clf_name == 'decition tree':
        
        clf = DecisionTreeClassifier(criterion= params['criterion'], min_samples_leaf= params['min_samples_leaf'], min_samples_split= params['min_samples_split'])

    elif clf_name == 'random forest' :
        clf = RandomForestClassifier(criterion= params['criterion'], min_samples_leaf= params['min_samples_leaf'], min_samples_split= params['min_samples_split'])
        
    elif clf_name == 'naive bayes' :
        clf = MultinomialNB(alpha= params['alpha'])

    elif clf_name == 'logistic regression':
        clf = LogisticRegression(penalty=params['penalty'])
    elif clf_name == 'AdaBoost':
        clf = AdaBoostClassifier(base_estimator=None,n_estimators=params['n_estimators'])
   
    return clf
       

clf = get_classifier(classifier_name, params)

# classification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
if clf:
    end = time.time()
    duration = end-start
    st.write('duration:%.3f second'%(duration))
    # "%d pens cost = %.2f" % (12, 150.87612)



# accuracy
accuracy_metric = accuracy_score(y_test,y_pred)
st.write(f'classifier = {classifier_name}')
st.write(f'accuracy = {accuracy_metric}')


#percision
percision_metric = precision_score(y_test,y_pred)
st.write(f'percision = {percision_metric}')

#recall
recall_metric = recall_score(y_test,y_pred)
st.write(f'recall = {recall_metric}')

#f1_score
f1_metric = f1_score(y_test,y_pred)
st.write(f'f1 = {f1_metric}')



if classifier_name == 'svm':
    fig = plt.figure()
    sns.set_style("whitegrid", {'axes.grid' : False})

    scores_train = clf.predict(X_train)
    scores_test = y_pred

    y_scores_train = []
    y_scores_test = []
    for i in range(len(scores_train)):
        y_scores_train.append(scores_train[i])

    for i in range(len(scores_test)):
        y_scores_test.append(scores_test[i])
        
    fpr_svm_train, tpr_svm_train, _ = roc_curve(y_train, y_scores_train, pos_label=1)
    fpr_svm_test, tpr_svm_test, _ = roc_curve(y_test, y_scores_test, pos_label=1)

    plt.plot(fpr_svm_train, tpr_svm_train, color='darkblue', label='Train AUC: %5f' %auc(fpr_svm_train, tpr_svm_train))
    plt.plot(fpr_svm_test, tpr_svm_test, color='red', ls='--', label='Test AUC: %5f' %auc(fpr_svm_test, tpr_svm_test))
    plt.title("SVM ROC Curve")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.legend(loc='lower right')
    st.pyplot(fig)

if classifier_name == 'decition tree':
    fig = plt.figure()
    sns.set(font_scale=1.5)
    sns.set_style("whitegrid", {'axes.grid' : False})

    scores_train = clf.predict(X_train)
    scores_test = y_pred

    y_scores_train = []
    y_scores_test = []
    for i in range(len(scores_train)):
        y_scores_train.append(scores_train[i])

    for i in range(len(scores_test)):
        y_scores_test.append(scores_test[i])
        
    fpr_dt_train, tpr_dt_train, _ = roc_curve(y_train, y_scores_train, pos_label=1)
    fpr_dt_test, tpr_dt_test, _ = roc_curve(y_test, y_scores_test, pos_label=1)

    plt.plot(fpr_dt_train, tpr_dt_train, color='black', label='Train AUC: %5f' %auc(fpr_dt_train, tpr_dt_train))
    plt.plot(fpr_dt_test, tpr_dt_test, color='red', ls='--', label='Test AUC: %5f' %auc(fpr_dt_test, tpr_dt_test))
    plt.title("Decision Tree ROC Curve")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.legend(loc='lower right')
    st.pyplot(fig)
if classifier_name == 'random forest':
    fig = plt.figure()
    sns.set_style("whitegrid", {'axes.grid' : False})

    scores_train = clf.predict(X_train)
    scores_test = y_pred

    y_scores_train = []
    y_scores_test = []
    for i in range(len(scores_train)):
        y_scores_train.append(scores_train[i])

    for i in range(len(scores_test)):
        y_scores_test.append(scores_test[i])
        
    fpr_rf_train, tpr_rf_train, _ = roc_curve(y_train, y_scores_train, pos_label=1)
    fpr_rf_test, tpr_rf_test, _ = roc_curve(y_test, y_scores_test, pos_label=1)

    plt.plot(fpr_rf_train, tpr_rf_train, color='darkblue', label='Train AUC: %5f' %auc(fpr_rf_train, tpr_rf_train))
    plt.plot(fpr_rf_test, tpr_rf_test, color='red', ls='--', label='Test AUC: %5f' %auc(fpr_rf_test, tpr_rf_test))
    plt.title("Random ForestROC Curve")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.legend(loc='lower right')
    st.pyplot(fig)

if classifier_name == 'naive bayes':
    fig = plt.figure()
    sns.set_style("whitegrid", {'axes.grid' : False})

    scores_train = clf.predict(X_train)
    scores_test = y_pred

    y_scores_train = []
    y_scores_test = []
    for i in range(len(scores_train)):
        y_scores_train.append(scores_train[i])

    for i in range(len(scores_test)):
        y_scores_test.append(scores_test[i])
        
    fpr_mnb_train, tpr_mnb_train, _ = roc_curve(y_train, y_scores_train, pos_label=1)
    fpr_mnb_test, tpr_mnb_test, _ = roc_curve(y_test, y_scores_test, pos_label=1)

    plt.plot(fpr_mnb_train, tpr_mnb_train, color='darkblue', label='Train AUC: %5f' %auc(fpr_mnb_train, tpr_mnb_train))
    plt.plot(fpr_mnb_test, tpr_mnb_test, color='red', ls='--', label='Test AUC: %5f' %auc(fpr_mnb_test, tpr_mnb_test))
    plt.title("Multinomial NB ROC Curve")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.legend(loc='lower right')
    st.pyplot(fig)

if classifier_name == 'logistic regression':
    fig = plt.figure()
    sns.set_style("whitegrid", {'axes.grid' : False})

    scores_train = clf.predict(X_train)
    scores_test = y_pred

    y_scores_train = []
    y_scores_test = []
    for i in range(len(scores_train)):
        y_scores_train.append(scores_train[i])

    for i in range(len(scores_test)):
        y_scores_test.append(scores_test[i])
        
    fpr_lr_train, tpr_lr_train, _ = roc_curve(y_train, y_scores_train, pos_label=1)
    fpr_lr_test, tpr_lr_test, _ = roc_curve(y_test, y_scores_test, pos_label=1)

    plt.plot(fpr_lr_train, tpr_lr_train, color='darkblue', label='Train AUC: %5f' %auc(fpr_lr_train, tpr_lr_train))
    plt.plot(fpr_lr_test, tpr_lr_test, color='red', ls='--', label='Test AUC: %5f' %auc(fpr_lr_test, tpr_lr_test))
    plt.title("LogisticRegression ROC Curve")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.legend(loc='lower right')
    st.pyplot(fig)






if classifier_name == 'AdaBoost':
    fig = plt.figure()
    sns.set_style("whitegrid", {'axes.grid' : False})

    scores_train = clf.predict(X_train)
    scores_test = y_pred

    y_scores_train = []
    y_scores_test = []
    for i in range(len(scores_train)):
        y_scores_train.append(scores_train[i])

    for i in range(len(scores_test)):
        y_scores_test.append(scores_test[i])
        
    fpr_lr_train, tpr_lr_train, _ = roc_curve(y_train, y_scores_train, pos_label=1)
    fpr_lr_test, tpr_lr_test, _ = roc_curve(y_test, y_scores_test, pos_label=1)

    plt.plot(fpr_lr_train, tpr_lr_train, color='darkblue', label='Train AUC: %5f' %auc(fpr_lr_train, tpr_lr_train))
    plt.plot(fpr_lr_test, tpr_lr_test, color='red', ls='--', label='Test AUC: %5f' %auc(fpr_lr_test, tpr_lr_test))
    plt.title("ADA Boost clf ROC Curve")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.legend(loc='lower right')
    st.pyplot(fig)


#plot
mertric_name = ['accuracy','percision','recall','f1']
metric_value = [accuracy_metric,percision_metric,recall_metric,f1_metric]
fig = plt.figure()
sns.barplot(x=mertric_name,y=metric_value)
st.pyplot(fig)




