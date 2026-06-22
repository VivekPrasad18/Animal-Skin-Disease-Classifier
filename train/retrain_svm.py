"""
Retrain SVM with GridSearch to find best params for all 6 diseases.
Uses train+valid combined for training, tests on test set.
"""
import os, warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

import numpy as np
import joblib
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(ROOT, 'dataset', 'train')
VAL_PATH   = os.path.join(ROOT, 'dataset', 'valid')
TEST_PATH  = os.path.join(ROOT, 'dataset', 'test')
MODELS_DIR = os.path.join(ROOT, 'models')

CLASS_NAMES = ['Dermatitis','Fungal_infections','Healthy',
               'Hypersensitivity','demodicosis','ringworm']

# ── Extract features ──────────────────────────────────────────────────────────
print('Loading ResNet50...')
resnet = ResNet50(weights='imagenet', include_top=False,
                  pooling='avg', input_shape=(224,224,3))

gen = ImageDataGenerator(preprocessing_function=preprocess_input)

print('Extracting train features...')
tr = gen.flow_from_directory(TRAIN_PATH, target_size=(224,224),
                              batch_size=32, class_mode='categorical', shuffle=False)
X_train = resnet.predict(tr, verbose=1); y_train = tr.classes

print('Extracting val features...')
vl = gen.flow_from_directory(VAL_PATH, target_size=(224,224),
                              batch_size=32, class_mode='categorical', shuffle=False)
X_val = resnet.predict(vl, verbose=1); y_val = vl.classes

print('Extracting test features...')
te = gen.flow_from_directory(TEST_PATH, target_size=(224,224),
                              batch_size=32, class_mode='categorical', shuffle=False)
X_test = resnet.predict(te, verbose=1); y_test = te.classes

# Combine train + val for more data
X_all = np.vstack([X_train, X_val])
y_all = np.concatenate([y_train, y_val])

# ── Scale ─────────────────────────────────────────────────────────────────────
scaler  = StandardScaler()
X_all   = scaler.fit_transform(X_all)
X_test  = scaler.transform(X_test)

# ── Grid Search ───────────────────────────────────────────────────────────────
print('\nRunning GridSearch (this takes a few minutes)...')
param_grid = {
    'C':     [1, 10, 50, 100],
    'gamma': [0.0001, 0.001, 0.01, 'scale'],
}
grid = GridSearchCV(
    SVC(kernel='rbf', class_weight='balanced', probability=True),
    param_grid, cv=3, scoring='f1_weighted', n_jobs=-1, verbose=2
)
grid.fit(X_all, y_all)

print('\nBest params:', grid.best_params_)
print('Best CV F1:', round(grid.best_score_ * 100, 2), '%')

# ── Final evaluation on test set ──────────────────────────────────────────────
best_svm = grid.best_estimator_
preds    = best_svm.predict(X_test)

print('\n===== TEST SET RESULTS =====')
print('Accuracy:', round(accuracy_score(y_test, preds) * 100, 2), '%')
print(classification_report(y_test, preds, target_names=CLASS_NAMES, zero_division=0))

# ── Save ──────────────────────────────────────────────────────────────────────
joblib.dump(best_svm, os.path.join(MODELS_DIR, 'svm_model.pkl'))
joblib.dump(scaler,   os.path.join(MODELS_DIR, 'scaler.pkl'))
print('Best SVM saved to models/')
