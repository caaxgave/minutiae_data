import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.compose import make_column_transformer
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import SCORERS
import matplotlib.pylab as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve,auc
from sklearn.model_selection import StratifiedKFold
import matplotlib.patches as patches
from sklearn.utils import resample

from xgboost import XGBClassifier

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import callbacks

from scipy.io.arff import loadarff

raw_data = loadarff("/content/minutiae_data/fingerprint_dataset.arff")
df_data = pd.DataFrame(raw_data[0])
X = df_data.copy()
y = X.pop('score')

"""### Preprocessing """

features_cat = ['fingerprint', 'minutia']
cols_names_num = df_data.drop(['fingerprint', 'minutia', 'score'], axis=1).columns

preprocessor = make_column_transformer(
    #(OneHotEncoder(handle_unknown='ignore'), features_cat),
    (OrdinalEncoder(), features_cat),
    (StandardScaler(), cols_names_num),
)

Encoder = LabelEncoder()

y = Encoder.fit_transform(y)
X = preprocessor.fit_transform(X)

"""## k-fold Cross-validation

### XGBoost
"""

cv = KFold(n_splits=15, random_state=7, shuffle=True) #K-Fold Function
model = XGBClassifier(n_estimators=250, learning_rate=0.1, random_state=7)

scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv)
#print("**** Classifier model: XGBoost ***")
#print('ROC-AUC: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
#print()
"""### Random Forest"""

clf = RandomForestClassifier(n_estimators= 500, criterion='entropy', random_state=7)
#clf = XGBClassifier(n_estimators=250, learning_rate=0.1, random_state=7)
cv = KFold(n_splits=5, random_state=7, shuffle=True)

fig1 = plt.figure(figsize=[6,6])
ax1 = fig1.add_subplot(111,aspect = 'equal')

tprs = []
aucs = []
mean_fpr = np.linspace(0,1,100)
i = 1
for train,test in cv.split(X,y):
    prediction = clf.fit(X[train],y[train]).predict_proba(X[test])
    fpr, tpr, t = roc_curve(y[test], prediction[:, 1])
    tprs.append(np.interp(mean_fpr, fpr, tpr))
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=2, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
    i= i+1

plt.plot([0,1],[0,1],linestyle = '--',lw = 2,color = 'black')
mean_tpr = np.mean(tprs, axis=0)
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, color='blue',
         label=r'Mean ROC (AUC = %0.2f )' % (mean_auc),lw=2, alpha=1)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for Random Forest')
plt.legend(loc="lower right")
#plt.show()
plt.savefig('RandomForest.png')

print()
"""### Neural Network"""

def create_model(input_shape):
  model = keras.Sequential()

  model.add( layers.Dense(units = 1000, 
                  input_dim=input_shape[0], 
                  activation='relu',) ) 
  
  model.add( layers.Dense(1000, activation='relu',) )
  model.add( layers.Dropout(0.3))
 
  model.add( layers.Dense(1000, activation='relu'))
 
  model.add( layers.BatchNormalization())
  model.add( layers.Dense(1000, activation='relu'))

  model.add( layers.Dense(1, activation='sigmoid',
                  ) )

  #compile
  model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
  )
  
  return model

cv = KFold(n_splits=10, random_state=1, shuffle=True)

fig1 = plt.figure(figsize=[6,6])
ax1 = fig1.add_subplot(111,aspect = 'equal')

tprs = []
aucs = []
mean_fpr = np.linspace(0,1,100)
i = 1

for train_index, valid_index in cv.split(X):
  model = create_model(X[train_index[1]].shape)

  early_stopping = keras.callbacks.EarlyStopping(
    patience=7,
    min_delta=0.001,
    restore_best_weights=True,
  )

  model.fit(   #history =
    X[train_index], y[train_index],
    validation_data=(X[valid_index], y[valid_index]),
    batch_size=400,
    epochs=50,
    callbacks=[early_stopping],
  )

  y_pred = model.predict(X[valid_index]).ravel()
  fpr, tpr, thresholds_keras = roc_curve(y[valid_index], y_pred)

  tprs.append(np.interp(mean_fpr, fpr, tpr))
  roc_auc = auc(fpr, tpr)
  aucs.append(roc_auc)
  plt.plot(fpr, tpr, lw=2, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
  i= i+1

plt.plot([0,1],[0,1],linestyle = '--',lw = 2,color = 'black')
mean_tpr = np.mean(tprs, axis=0)
mean_auc_nn = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, color='blue',
         label=r'Mean ROC (AUC = %0.2f )' % (mean_auc_nn),lw=2, alpha=1)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for Neural Network')
plt.legend(loc="lower right")
#plt.show()
plt.savefig('NeuralNetwork.png')

print()
print()
print("**** Classifier model: XGBoost ****")
print('ROC-AUC: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
print()
print("**** Classifier model: Random Forest ****")
print('ROC-AUC: %.3f' % mean_auc)
print()
print("**** Classifier model: Neural Network ****")
print("ROC-AUC: %.3f" % mean_auc_nn)
