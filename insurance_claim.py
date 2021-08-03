import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn import metrics
from sklearn.linear_model import LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from sklearn import preprocessing
from sklearn.datasets import make_classification
import tensorflow as tf
from tensorflow import keras
from matplotlib.widgets import Slider
from tensorboard.plugins.hparams import api as hp
import math
import seaborn as sns

accuracy_y = []
name_x = []
acc_1 =  []
acc_2 = []
def load_dataset():
    insurance_csv = pd.read_csv('insurance_claims.csv')
    insurance_csv = insurance_csv.drop(['policy_bind_date' , 'incident_date','incident_location'], axis = 1)
    return insurance_csv

def data_description(insurance_csv):
    print('Number of Entries: ',insurance_csv.shape[0])
    print('Number of features: ',insurance_csv.shape[1])
    print('All features of data: ',list(insurance_csv.columns))
    print('No of fraud and non fraud policies: ',insurance_csv.fraud_reported.value_counts())
    print('\n\n\n')
    print('Now showing unique meta-data ')
    print('\n')
    for col in insurance_csv.columns:
        print(col,' meta data :  ',insurance_csv[col].unique())

    #columns having Object data type
    #features who needs scaling
    col_ob_dt = [col for col in insurance_csv.columns if insurance_csv[col].dtype == 'O']
    print('Columns having ob datatype : ',col_ob_dt)
    return col_ob_dt

def data_visualization(col_ob_dt,insurance_csv):
    #plotting catagorial data vs it's count
    sns.pairplot(insurance_csv)
    plt.show()
    fig = plt.figure()

    #Now plotting meta data of catagorical data
    #4 graphs per page for much clearer appreance
    #deviding whole catagoriacal data into 5 parts
    print('size       ',len(col_ob_dt))
    col_ob_dt_mat = []
    col_ob_dt_mat.append( col_ob_dt[ : 4])
    col_ob_dt_mat.append( col_ob_dt[4  : 8])
    col_ob_dt_mat.append( col_ob_dt[8 : 12])
    col_ob_dt_mat.append( col_ob_dt[12  : 16] )
    col_ob_dt_mat.append( col_ob_dt[16  : ])

    #plotting all the parts 
    #plotting all catagorial data against their count as bar plot
    for j in range(5):
        count = 0
        for i in col_ob_dt_mat[j]:
            ax=plt.subplot(2,2,count + 1) 
            plt.title(i)
            count = count + 1
            pd.value_counts(insurance_csv[i]).plot.bar()
        plt.tight_layout()
        plt.show()

    #plotting all catagorial data against their fraud detection count as stacked bar plot
    for j in range(5):
        count = 0
        for i in col_ob_dt_mat[j]:
            ax=plt.subplot(2,2,count + 1) 
            plt.title(i)
            count = count + 1
            table = insurance_csv.pivot_table(index=insurance_csv[i], columns='fraud_reported', aggfunc='size')
            table.plot(ax=ax,kind='bar', stacked=True)
        plt.tight_layout()
        plt.show()

def encode_data(insurance_csv,col_ob_dt):
    le = preprocessing.LabelEncoder()
    for column in col_ob_dt:
        insurance_csv[column] = le.fit_transform(insurance_csv[column].astype(str))
        print(column,'         ',le.classes_)

    print("----------------------------------------------------------------------")
    insurance_csv.to_csv('Sample.csv')

def preprocess_data(insurance_csv,col_ob_dt):
    col_to_normalize = list( set(insurance_csv.columns) -  set(col_ob_dt))
    insurance_csv[col_to_normalize] = preprocessing.normalize(insurance_csv[col_to_normalize])
    insurance_csv.to_csv('Sample1.csv')
    return insurance_csv

def train_test_splitting(insurance_csv_orig):
    insurance_csv = insurance_csv_orig
    insurance_csv_x = insurance_csv.drop('fraud_reported' , axis = 1)
    insurance_csv_y = insurance_csv_orig['fraud_reported']
    X_train, X_test, y_train, y_test = train_test_split(insurance_csv_x, insurance_csv_y, test_size=0.3)
    return X_train, X_test, y_train, y_test

def reshapping(X_train, X_test, y_train, y_test):
    y_test = np.array(y_test)
    y_train = np.array(y_train)
    y_test = y_test.reshape(y_test.shape[0],1)
    y_train = y_train.reshape(y_train.shape[0],1)
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    print (X_train.shape,'   ' ,y_train.shape)
    print (X_test.shape,'   ' ,y_test.shape)
    print(type(X_train),'  ',type(X_test),'  ',type(y_train),'  ',type(y_test))
    return X_train, X_test, y_train, y_test

def info_dataset_after_preprocess( X_train,X_test, y_train,  y_test):
    print('before Dim change')
    print('Number of training examples: m_train = ',X_train.shape[0])
    print('Number of testing examples: m_test = ',X_test.shape[0])
    print('train x shape ', X_train.shape)
    print('train y shape', y_train.shape)
    print('test x shape ', X_test.shape)
    print('test y shape', y_test.shape)
    print('---------------------------------------')

def logistic_regression(X_train, X_test, y_train, y_test):
    model = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial',  max_iter=10000).fit(X_train, y_train.ravel())
    model.predict(X_test)
    model.predict_proba(X_test) 
    print('------------------')
    y_pred = model.predict(X_test)
    print('logistic_regression    ',accuracy_score(y_test,y_pred)*100)
    accuracy_y.append(accuracy_score(y_test,y_pred)*100)
    name_x.append('logistic_regression')
    #plot_classification_report(classification_report(y_test, y_pred),'logistic_regression')

def logistic_regression_CV(X_train, X_test, y_train, y_test):
    model = LogisticRegressionCV(cv = 5,random_state=0, multi_class='multinomial',  max_iter=10000).fit(X_train, y_train.ravel())
    model.predict(X_test)
    model.predict_proba(X_test) 
    print('------------------')
    y_pred = model.predict(X_test)
    print('logistic_regression _CV   ',accuracy_score(y_test,y_pred)*100)
    accuracy_y.append(accuracy_score(y_test,y_pred)*100)
    name_x.append('logistic_regression_CV')
    #plot_classification_report(classification_report(y_test, y_pred),'logistic_regression_CV')

def  DecisionTree_Classifier(X_train, X_test, y_train, y_test):
    clf = DecisionTreeClassifier(random_state=0, max_depth=3, min_samples_leaf=4)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(' DecisionTree_Classifier accuracy  :', accuracy_score(y_test,y_pred)*100)
    accuracy_y.append(accuracy_score(y_test,y_pred)*100)
    name_x.append('DecisionTree_Classifier')
    #plot_classification_report(classification_report(y_test, y_pred),'DecisionTree_Classifier')

def gaussian_NB(X_train, X_test, y_train, y_test):
    model = GaussianNB().fit(X_train, y_train.ravel())
    model.predict(X_test)
    model.predict_proba(X_test) 
    y_pred = model.predict(X_test)
    print('gaussian_NB    ',accuracy_score(y_test,y_pred)*100)
    accuracy_y.append(accuracy_score(y_test,y_pred)*100)
    name_x.append('gaussian_NB')
    #plot_classification_report(classification_report(y_test, y_pred),'gaussian_NB')

def SVC_classifire(X_train, X_test, y_train, y_test):
    clf = SVC(cache_size=200, random_state= 0 )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(' SVC_Classifier accuracy  :', accuracy_score(y_test,y_pred)*100)
    accuracy_y.append(accuracy_score(y_test,y_pred)*100)
    name_x.append('SVC_classifire')
    #plot_classification_report(classification_report(y_test, y_pred),'SVC')

def RandomForest_Classifier(X_train, X_test, y_train, y_test):
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(' RandomForest_Classifier accuracy  :', accuracy_score(y_test,y_pred)*100)
    accuracy_y.append(accuracy_score(y_test,y_pred)*100)
    name_x.append('RandomForest_Classifier')
    #plot_classification_report(classification_report(y_test, y_pred),'RandomForestClassifier')

def GaussianProcess_Classifier(X_train, X_test, y_train, y_test):
    clf = GaussianProcessClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(' GaussianProcess_Classifier accuracy  :', accuracy_score(y_test,y_pred)*100)
    accuracy_y.append(accuracy_score(y_test,y_pred)*100)
    name_x.append('GaussianProcess_Classifier')
    #plot_classification_report(classification_report(y_test, y_pred),'GaussianProcess_Classifier')

def show_values(pc, fmt="%.2f", **kw):

    from itertools import zip_longest as zip
    pc.update_scalarmappable()
    ax = pc.axes
    for p, color, value in zip(pc.get_paths(), pc.get_facecolors(), pc.get_array()):
        x, y = p.vertices[:-2, :].mean(0)
        if np.all(color[:3] > 0.5):
            color = (0.0, 0.0, 0.0)
        else:
            color = (1.0, 1.0, 1.0)
        ax.text(x, y, fmt % value, ha="center", va="center", color=color, **kw)

def cm2inch(*tupl):

    inch = 2.54
    if type(tupl[0]) == tuple:
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)

def heatmap(AUC, title, xlabel, ylabel, xticklabels, yticklabels, figure_width=40, figure_height=20, correct_orientation=False, cmap='RdBu'):


    # Plot it out
    fig, ax = plt.subplots()    
    #c = ax.pcolor(AUC, edgecolors='k', linestyle= 'dashed', linewidths=0.2, cmap='RdBu', vmin=0.0, vmax=1.0)
    c = ax.pcolor(AUC, edgecolors='k', linestyle= 'dashed', linewidths=0.2, cmap=cmap)

    # put the major ticks at the middle of each cell
    ax.set_yticks(np.arange(AUC.shape[0]) + 0.5, minor=False)
    ax.set_xticks(np.arange(AUC.shape[1]) + 0.5, minor=False)

    # set tick labels
    #ax.set_xticklabels(np.arange(1,AUC.shape[1]+1), minor=False)
    ax.set_xticklabels(xticklabels, minor=False)
    ax.set_yticklabels(yticklabels, minor=False)

    # set title and x/y labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)      

    # Remove last blank column
    plt.xlim( (0, AUC.shape[1]) )

    # Turn off all the ticks
    ax = plt.gca()    
    for t in ax.xaxis.get_major_ticks():
        t.tick1line.set_visible(False)
        t.tick1line.set_visible(False)
    for t in ax.yaxis.get_major_ticks():
        t.tick1line.set_visible(False)
        t.tick1line.set_visible(False)

    # Add color bar
    plt.colorbar(c)

    # Add text in each cell 
    show_values(c)

    # Proper orientation (origin at the top left instead of bottom left)
    if correct_orientation:
        ax.invert_yaxis()
        ax.xaxis.tick_top()       

    # resize 
    fig = plt.gcf()
    #fig.set_size_inches(cm2inch(40, 20))
    #fig.set_size_inches(cm2inch(40*4, 20*4))
    fig.set_size_inches(cm2inch(figure_width, figure_height))
    title = str(title)
    title = ''.join(e for e in title if e.isalnum())
    print(title)
    #plt.show()

def plot_classification_report(classification_report, title='Classification report ', cmap='RdBu'):

    lines = classification_report.split('\n')

    classes = []
    plotMat = []
    support = []
    class_names = []
    for line in lines[2 : (len(lines) - 4)]:
        t = line.strip().split()
        if len(t) < 2: continue
        classes.append(t[0])
        v = [float(x) for x in t[1: len(t) - 1]]
        support.append(int(t[-1]))
        class_names.append(t[0])
        print(v)
        plotMat.append(v)

    print('plotMat: {0}'.format(plotMat))
    print('support: {0}'.format(support))

    xlabel = 'Metrics'
    ylabel = 'Classes'
    xticklabels = ['Precision', 'Recall', 'F1-score']
    yticklabels = ['{0} ({1})'.format(class_names[idx], sup) for idx, sup  in enumerate(support)]
    figure_width = 25
    figure_height = len(class_names) + 7
    correct_orientation = False
    heatmap(np.array(plotMat), title, xlabel, ylabel, xticklabels, yticklabels, figure_width, figure_height, correct_orientation, cmap=cmap)

def train_test_model(hparams,X_train,X_test, y_train,  y_test):
  model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(hparams['num_units_1'], activation=tf.nn.relu),
    tf.keras.layers.Dropout(hparams['dropout_1']),
    #tf.keras.layers.Dense(hparams['num_units_2'],kernel_regularizer=keras.regularizers.l2(0.1), activation=tf.nn.relu),
    tf.keras.layers.Dense(hparams['num_units_2'],activation=tf.nn.relu),
    tf.keras.layers.Dropout(hparams['dropout_2']),
    tf.keras.layers.Dense(1, activation=tf.nn.sigmoid),
    tf.keras.layers.Flatten()
  ])
  model.compile(
      optimizer=hparams['optimizer'],
      loss='binary_crossentropy',
      metrics=['accuracy'],
      learning_rate= 1e-7 ,
  )

  model.fit(X_train, y_train, epochs=100) 
  _, accuracy = model.evaluate(X_test, y_test)
  print( model.evaluate(X_test, y_test))
  y_pred = model.predict(X_test)


  y_pred =(y_pred>0.5)
  results = confusion_matrix(y_test, y_pred) 
  print ('Confusion Matrix :')
  print(results) 

  #report
  s = ('ANN with ',str(hparams))

  print (classification_report(y_test, y_pred) )
  #accuracy_y.append(accuracy_score(y_test,y_pred)*100)
  s = ' '.join(e for e in s if e.isalnum())
  #name_x.append(s)
  #plot_classification_report(classification_report(y_test, y_pred),s)

  if hparams['optimizer'] == 'sgd':
      acc_1.append(accuracy)
  else :
      acc_2.append(accuracy)
  return accuracy

def run(run_dir, hparams,X_train, X_test, y_train, y_test):
    #hp.hparams(hparams)  # record the values used in this trial
    accuracy = train_test_model(hparams,X_train, X_test, y_train, y_test)
    tf.summary.scalar('accuracy', accuracy)

def ANN_with_gridSearch(X_train, X_test, y_train, y_test):
    HP_NUM_UNITS_1 = hp.HParam('num_units_1', hp.Discrete(list(range(32,33))))
    HP_DROPOUT_1 = hp.HParam('dropout_1', hp.RealInterval(0.3, 0.4))
    HP_NUM_UNITS_2 = hp.HParam('num_units_2', hp.Discrete(list(range(16,17))))
    HP_DROPOUT_2 = hp.HParam('dropout_2', hp.RealInterval(0.3, 0.4))
    HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam','sgd']))
    METRIC_ACCURACY = 'accuracy'
    hparams=[HP_NUM_UNITS_1, HP_DROPOUT_1,HP_NUM_UNITS_2, HP_DROPOUT_2, HP_OPTIMIZER]
    hparams=[HP_NUM_UNITS_1,HP_NUM_UNITS_2,  HP_OPTIMIZER]
    metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')]
   
    session_num = 0

    for num_units_1 in HP_NUM_UNITS_1.domain.values:
      for dropout_rate_1 in (HP_DROPOUT_1.domain.min_value, HP_DROPOUT_1.domain.max_value):
          for num_units_2 in HP_NUM_UNITS_2.domain.values:
            for dropout_rate_2 in (HP_DROPOUT_2.domain.min_value, HP_DROPOUT_1.domain.max_value):
                for optimizer in HP_OPTIMIZER.domain.values:
                  hparamss = {
                      HP_NUM_UNITS_1: num_units_1,
                      HP_DROPOUT_1: dropout_rate_1,
                      HP_NUM_UNITS_2: num_units_2,
                      HP_DROPOUT_2: dropout_rate_2,
                      HP_OPTIMIZER: optimizer,
                  }
                  run_name = "run-%d" % session_num
                  print('--- Starting trial: %s' % run_name)
                  print({h.name: hparamss[h] for h in hparamss})
                  run('logs/hparam_tuning/' + run_name, {h.name: hparamss[h] for h in hparamss},X_train, X_test, y_train, y_test)
                  session_num += 1

def main():
    insurance_csv = load_dataset()
    col_ob_dt = data_description(insurance_csv)
    #data_visualization(col_ob_dt,insurance_csv)
    encode_data(insurance_csv,col_ob_dt)
    insurance_csv_norm = preprocess_data(insurance_csv,col_ob_dt)
    X_train, X_test, y_train, y_test = train_test_splitting(insurance_csv_norm)
    X_train, X_test, y_train, y_test = reshapping(X_train, X_test, y_train, y_test)
    info_dataset_after_preprocess(X_train, X_test, y_train, y_test)
    logistic_regression(X_train, X_test, y_train, y_test)
    logistic_regression_CV(X_train, X_test, y_train, y_test)
    gaussian_NB(X_train, X_test, y_train, y_test)
    #ANN_with_gridSearch(X_train, X_test, y_train, y_test)
    #plt.plot(acc_1)
    #plt.plot(acc_2)
    ##plt.show()
    #corrmat = insurance_csv_norm.corr()
    #sns.set(font_scale=1.00)
    #sns.heatmap(corrmat,cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 5})
    #plt.show()
    DecisionTree_Classifier(X_train, X_test, y_train, y_test)
    SVC_classifire(X_train, X_test, y_train, y_test)
    RandomForest_Classifier(X_train, X_test, y_train, y_test)
    GaussianProcess_Classifier(X_train, X_test, y_train, y_test)

    plt.plot(name_x,accuracy_y)
    plt.xlabel("Classifire ")
    plt.ylabel("Accuracy % ")
    plt.show()

if __name__ ==   "__main__":
    main()
