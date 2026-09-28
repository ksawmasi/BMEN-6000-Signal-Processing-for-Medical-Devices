import pandas as pd
import numpy as np
import wfdb
import ast
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Normalization
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix


# -------------------------
# Setup: data formatting, loading, storage
# function for importing data
def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data

# You will need to change this to wherever you downloaded the data to
path = 'C:/Users/alima/documents/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/'
sampling_rate=100

#number of classes = 5: 
    #'NORM' = normal ECG; 
    #'MI' = Myocardial infarction; 
    #'CD' = conduction disturbance;
    #'STTC' = ST/T Change;
    #'HYP' = hypertrophy
classes = ['NORM', 'MI', 'CD', 'STTC', 'HYP']

# load and convert annotation data
Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

# Load raw signal data using function defined at top
X = load_raw_data(Y, sampling_rate, path)

# Load scp_statements.csv for diagnosis
agg_df = pd.read_csv(path+'scp_statements.csv', index_col=0)
agg_df = agg_df[agg_df.diagnostic == 1]

def aggregate_diagnostic(y_dic):
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)
    return list(set(tmp))

# Appending diagnosis to data frame Y 
Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)

# Code diagnoses as an array of binary values
Z = pd.DataFrame(0, index = Y.index, columns = classes, dtype = 'int')
for i in Z.index:
    for k in Y.loc[i].diagnostic_superclass:
        Z.loc[i, k] = 1

# Calculate statistics here
# ------------------------------

# Loop through all the classes
statistics = []

for diagnosis in classes:
    # find all records
    diagnosis_mask = Z[diagnosis] == 1
    # find corresponding rows
    diagnosis_data = Y.loc[diagnosis_mask]
    # number of cases
    num_cases = diagnosis_mask.sum()
    # average age
    mean_age = diagnosis_data['age'].mean()
    # standard deviation of age
    std_age = diagnosis_data['age'].std()
    # average weight
    mean_weight = diagnosis_data['weight'].mean()
    # standard deviation of weight
    std_weight = diagnosis_data['weight'].std()
    # male count
    male_count = (diagnosis_data['sex'] == 0).sum()
    # female count
    female_count = (diagnosis_data['sex'] == 1).sum()
    # store stats
    statistics.append([
        diagnosis,
        num_cases,
        mean_age,
        std_age,
        mean_weight,
        std_weight,
        male_count,
        female_count
    ])
# put stats in dataframe
    statistics_add_df = pd.DataFrame(
        statistics,
        columns=[
            'Diagnosis',
            'Number of Cases',
            'Average Age',
            'Age Standard Deviation',
            'Average Weight',
            'Weight Standard Deviation',
            'Male',
            'Female'
        ]
    )
    
# ------------------------------
# Split data into train and test

# strat_fold is a random number 1-10 assigned by the creators of the dataset specifically for the purpose of splitting data into test and train datasets.
# Condition 1: 10/90
# test_fold_max = 1
# Condition 2,4,5: 50/50
# test_fold_max = 5
# Condition 3: 90/10
test_fold_max = 9

# Weight metadata
meta_data = np.array([Y.weight]) # fill in your choice of metadata here; you can copy/paste more copies below to add more variables
meta_data = np.repeat(meta_data, 1000, axis = 0)
meta_data = meta_data.T
X = np.dstack((X,meta_data))
# Remove NaN and assign value = 0
X = np.nan_to_num(X) 
# Normalize ECG data and metadata (otherwise orders of magnitude difference in size)
normalize_layer = Normalization(axis=-1)
normalize_layer.adapt(X)

# Training data
X_train = X[np.where(Y.strat_fold <= test_fold_max)]
y_train = Z[(Y.strat_fold <= test_fold_max)]
# Test data
X_test = X[np.where(Y.strat_fold > test_fold_max)]
y_test = Z[Y.strat_fold > test_fold_max]

# Create tensor dataset from raw data
dataset = tf.data.Dataset.from_tensor_slices((X_train,y_train)).batch(1)
eval_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(1)



# ------------------------------
# Model 

# Create a model with three layers: 1D CNN; 1D CNN; Dense
model = keras.Sequential([
    normalize_layer,
    Conv1D(64, 5, activation='relu'),
    MaxPooling1D(2),   
    Conv1D(64, 3, activation='relu'),
    MaxPooling1D(2),
    # Data needs to be "flattened" before running through a dense NN
    Flatten(),               
    # Final layer needs to match the shape of the output. 
    # Sigmoid activation is for binary (Y/N) outputs
    Dense(Z.shape[-1], activation="sigmoid")               
])

# Tell the model what to optimize for
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy', 'Precision', 'Recall'])

# Run
model.fit(dataset, epochs=10)

# Report out metrics from model.compile; binary_accuracy is raw overall accuracy
accuracy_score = model.evaluate(eval_dataset, return_dict=True)

# Make predictions for the test data
predictions = model.predict(X_test)

# Find the locations of maximum certainty
max_locations = np.argmax(predictions, axis=1)
predicted_classes_max = np.zeros(predictions.shape).astype(int)
length = predictions.shape[0]
for i in range(length):
    predicted_classes_max[i,max_locations[i]] = 1

# Also find locations where certainty > threshold (to account for multiple diagnoses)
# You can change this threshold if you like
predicted_classes_thresh = (predictions > 0.6).astype(int)

# Union of "best" and "all other good" prediction matrices
all_predictions = predicted_classes_max | predicted_classes_thresh

# Confusion matrix
mlcm = multilabel_confusion_matrix(y_test, all_predictions)

# Condition Results
results_cond = []

for i in range(len(classes)):

    tn, fp, fn, tp = mlcm[i].ravel()

    binary_accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    results_cond.append([
        classes[i],
        tp,
        tn,
        fp,
        fn,
        binary_accuracy,
        sensitivity,
        specificity
    ])

results_cond_df = pd.DataFrame(
    results_cond,
    columns=[
        'Diagnosis',
        'TP',
        'TN',
        'FP',
        'FN',
        'Overall Accuracy',
        'Sensitivity',
        'Specificity'
    ]
)

# confusion matrix dataframes
NORM_df = pd.DataFrame(
    mlcm[0],
    index=['True Negative', 'True Positive'],
    columns=['False Negative', 'False Positive']
)

MI_df = pd.DataFrame(
    mlcm[1],
    index=['True Negative', 'True Positive'],
    columns=['False Negative', 'False Positive']
)

CD_df = pd.DataFrame(
    mlcm[2],
    index=['True Negative', 'True Positive'],
    columns=['False Negative', 'False Positive']
)

STTC_df = pd.DataFrame(
    mlcm[3],
    index=['True Negative', 'True Positive'],
    columns=['False Negative', 'False Positive']
)

HYP_df = pd.DataFrame(
    mlcm[4],
    index=['True Negative', 'True Positive'],
    columns=['False Negative', 'False Positive']
)


