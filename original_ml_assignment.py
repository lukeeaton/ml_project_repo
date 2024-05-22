# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 22:26:46 2023

@author: 260690
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import pandas as pd
import seaborn as sns
import random
from itertools import cycle
from typing import Callable
from statistics import mean

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_fscore_support, classification_report, recall_score, roc_curve, auc
from sklearn.preprocessing import label_binarize

seed = 42

#%% Loading data
filename = "data/Dry_Bean_Dataset.csv"

with open(filename) as f:
    reader = csv.reader(f)
    
    # Check features & pass header row
    header_row = next(reader)
    column_titles = {}
    for index, column_header in enumerate(header_row):
        column_titles[index] = column_header
        print(index, column_header)   
    
    # Bring data in from file
    data_as_list = []
    for row in reader:
        data_as_list.append([value for value in row])
    data = np.array(data_as_list)    
    
df = pd.read_csv(filename) # Pandas dataframe for easier/quick data checking

#%% Inspect & clean data
# Fixing feature names
# print(column_titles)
feature_names = ['Area', 'Perimeter', 'Major Axis Length', 'Minor Axis Length',
                 'Aspect Ratio', 'Eccentricity', 'Convex Area',
                 'Equivalent Diameter', 'Extent', 'Solidity', 'Roundness',
                 'Compactness', 'Shape Factor 1', 'Shape Factor 2',
                 'Shape Factor 3', 'Shape Factor 4']

bean_classes = ['Barbunya', 'Bombay', 'Cali', 'Dermason', 'Horoz', 'Seker', 'Sira']

for i in range(len(feature_names)): # <-- Just for pandas dataframe
    df.rename(columns={column_titles[i]: feature_names[i]}, inplace=True)
  
# head = data[:10]
# print(head)

print(f"\nShape of the dataset: {str(data.shape)}")


# Get descriptive statistics to see if any feature datatypes need to be changed
print("\n Feature descriptive statistics:")
data_descriptions = df.describe(percentiles=[.25, .5, .75, .995]).T.round(2)
print(data_descriptions)

print("\n Information about feature datatypes:") # <-- Must convert 0:Area & 6:Convex Area into float.
print(df.info())

data_features = data[:, np.arange(0, 16)] # <-- Everything to do with 'data' is for the numpy array
data_features = data_features.astype(float)

data_labels = data[:, 16]
print(f"\n The original labels were: {str(np.unique(data[:, 16]))}")


# Recode labels from names to numbers
np.place(data_labels, data_labels=='BARBUNYA', '0')
np.place(data_labels, data_labels=='BOMBAY', '1')
np.place(data_labels, data_labels=='CALI', '2')
np.place(data_labels, data_labels=='DERMASON', '3')
np.place(data_labels, data_labels=='HOROZ', '4')
np.place(data_labels, data_labels=='SEKER', '5')
np.place(data_labels, data_labels=='SIRA', '6')
data_labels = data_labels.astype(int)
print(f"\n The recoded labels are: {str(np.unique(data_labels))}")

print(f"\n An indexed list of the original column titles (Features): {str(column_titles)}")
print(f"\n A peek at the dataset feature values: {str(data_features)}")
print(f"\n A peek at the dataset labels: {str(data_labels)}")


# Checking for & removing duplicates
# duplicates = df.duplicated(subset=None, keep='first').sum()
# print(f"\n There are {str(duplicates)} duplicates.") <-- Alternative quick test using pandas df
n_data_items = data.shape[0]
n_unique_data_items = (np.unique(data, axis=0).shape[0])
n_duplicate_data_items = n_data_items - n_unique_data_items
print(f"\n There are {n_data_items} total items.")
print(f"\n There are {n_unique_data_items} unique items.")
print(f"\n So there are {n_duplicate_data_items} duplicate data items... Removing duplicates...")
data = np.unique(data, axis=0).shape[0]


# Check for missing data
missing_data = 0
for i in range(data_features.shape[1]):  
    for x in range(data_features.shape[0]):
        if data_features[x, i]  == 0:
            missing_data += 1
        
print(f"\n There are {missing_data} instances of missing data.") # <-- Confirms what the pandas df.info says above.

#%% Visualise data
# Quick, easy histograms to visualise
sns.set(style='whitegrid')
sns.countplot(x='Class', data=df)
print(df['Class'].value_counts())

# Histogram for distribution of instances by outcome class
def bean_dist_plot(plot_data, plot_title):
    plt.figure()
    _, _, _ = plt.hist(plot_data, bins=[0, 1, 2, 3, 4, 5, 6, 7], align='left', color='grey')
    plt.title(plot_title, fontname='Times New Roman')
    plt.xticks(np.unique(plot_data), bean_classes, rotation=45, fontname='Times New Roman')
    plt.xlim(left=min(np.unique(plot_data))-1, right=max(np.unique(plot_data))+1)
    plt.xlabel('Value', fontname='Times New Roman')
    plt.ylabel('Frequency', fontname='Times New Roman')
    plt.show()
    print("\n")

bean_dist_plot(data_labels, "Figure 1: Bean class frequencies for the full dataset")


# Boxplots for the features
def feature_box_plot(feature_col, outcome_groups, plot_title):
    plt.figure()
    groups = np.unique(outcome_groups)
    plot_data = []
    for i in groups:
        plot_data.append(feature_col[outcome_groups==i])
    plt.boxplot(plot_data)
    plt.title(plot_title, fontname='Times New Roman')
    plt.ylabel("Value", fontname='Times New Roman')
    plt.xlabel("Bean Class", fontname='Times New Roman')
    plt.xticks(groups+1, bean_classes)
    plt.show()
    print("\n")

for feature in range(data_features.shape[1]):
    feature_box_plot(data_features[:, feature], data_labels, "Boxplots for "+str(feature_names[feature]))


# Histograms for distributions of values by feature:
def feature_histogram(feature_col, plot_title):
    plt.figure()
    plt.hist(feature_col, bins=30, color='grey')
    plt.title("Distribution of values for feature: "+str(plot_title), fontname='Times New Roman')
    plt.ylabel("Frequency", fontname='Times New Roman')
    plt.xlabel(str(plot_title)+" (pixel count)")
    plt.xlabel('Value', fontname='Times New Roman')
    plt.ylabel('Frequency', fontname='Times New Roman')
    plt.axvline(x=feature_col.mean(), color='black', label='Mean', linestyle='dotted', linewidth=2)
    plt.show()
    
for feature in range(data_features.shape[1]):
    feature_histogram(data_features[:, feature], str(feature_names[feature]))


# Pearson's Correlation matrix
def corr_plot(data):
    plt.figure(figsize=(15,15))
    plt.title("Figure 4: Correlation matrix", fontname='Times New Roman', fontsize=30)
    sns.heatmap(data.corr("pearson"), cmap='coolwarm', annot=True)

corr_plot(df[feature_names])
deleted_ids = []
#%% Remove outliers; Drop features with high correlations
all_ids = np.arange(0, data_features.shape[0])
deleted_ids = []

for i in range(all_ids.shape[0]):
    if data_features[i, 5] < 0.25:
        print(f"Deleted an outlier in the feature: {feature_names[5]}")
        deleted_ids.append(i)
    if data_features[i, 9] < 0.93:
        print(f"Deleted an outlier in the feature: {feature_names[9]}")
        deleted_ids.append(i)
    if data_features[i, 10] < 0.54:
        print(f"Deleted an outlier in the feature: {feature_names[10]}")
        deleted_ids.append(i)
    if data_features[i, 15] < 0.953:
        print(f"Deleted an outlier in the feature: {feature_names[15]}")
        deleted_ids.append(i)

data_features = np.delete(data_features, deleted_ids, axis=0)
data_labels = np.delete(data_labels, deleted_ids, axis=0)

print(f"Deleted the instances with ids: {deleted_ids}")

data_features = np.delete(data_features, [6, 14], axis=1) # Slimmed A - drop perfect correlations (Option B from write up)
# data_features = np.delete(data_features, [0, 1, 6, 7, 11, 14], axis=1) # Slimmed B - drop > 95% correlation (Option C from write up)
# data_features = np.delete(data_features, [0, 1, 6, 7, 11, 12, 14], axis=1) # Slimmed C - drop >= 95% correlation (Option D from write up)

#%% Splitting the data into training, testing, and validation sets
all_ids = np.arange(0, data_features.shape[0])
full_train_ids, test_ids = train_test_split(all_ids, test_size=0.2, train_size=0.8,
                                       random_state=seed, shuffle=True,
                                       stratify=data_labels)

train_ids, val_ids = train_test_split(full_train_ids, test_size=0.1, train_size=0.9,
                                       random_state=seed, shuffle=True,
                                       stratify=data_labels[full_train_ids])


# Distributions of the classes in the training, testing, and validation sets - checking that they are similar
bean_dist_plot(data_labels[train_ids], "Bean class frequencies for the training dataset")
bean_dist_plot(data_labels[test_ids], "Bean class frequencies for the testing dataset")
bean_dist_plot(data_labels[val_ids], "Bean class frequencies for the validation dataset")
# ^^ Confirms that they are stratified effectively - have similar distributions.

#%% Scaling the feature data
def scale_features(features):
    scaler = StandardScaler()
    scaler.fit(features)
    scaled_features = scaler.transform(features)
    print(f"\n A peek at the scaled dataset features:\n {scaled_features}")
    
    return scaled_features

scaled_data_features = scale_features(data_features)

#%% Creating the model classes (Code influenced by labs)
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
class two_hidden_layer_MLP(nn.Module): 
    def __init__(self,
                 input_size: int,
                 hidden_layer_sizes: list,
                 output_size: int,
                 activation_fn: Callable[[torch.Tensor], torch.Tensor] = torch.sigmoid):
        super().__init__()
        self.hidden_l1 = nn.Linear(input_size, hidden_layer_sizes[0])
        self.hidden_l2 = nn.Linear(hidden_layer_sizes[0], hidden_layer_sizes[1])
        self.output_l3 = nn.Linear(hidden_layer_sizes[1], output_size)
        self.activation_fn = activation_fn
        
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.hidden_l1(inputs)
        x = self.hidden_l2(x)
        x = self.output_l3(x)
        x = self.activation_fn(x)
        return(x)
    
    
class three_hidden_layer_MLP(nn.Module): # <-- This is the one used in the final model
    def __init__(self,
                 input_size: int,
                 hidden_layer_sizes: list,
                 output_size: int,
                 activation_fn: Callable[[torch.Tensor], torch.Tensor] = F.relu):
        super().__init__()
        self.hidden_l1 = nn.Linear(input_size, hidden_layer_sizes[0])
        self.hidden_l2 = nn.Linear(hidden_layer_sizes[0], hidden_layer_sizes[1])
        self.hidden_l3 = nn.Linear(hidden_layer_sizes[1], hidden_layer_sizes[2])
        self.output_l4 = nn.Linear(hidden_layer_sizes[2], output_size)
        self.activation_fn = activation_fn
        
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.hidden_l1(inputs)
        x = self.hidden_l2(x)
        x = self.hidden_l3(x)
        x = self.output_l4(x)
        x = self.activation_fn(x)
        return(x)
  
    
class four_hidden_layer_MLP(nn.Module): 
    def __init__(self,
                 input_size: int,
                 hidden_layer_sizes: list,
                 output_size: int,
                 activation_fn: Callable[[torch.Tensor], torch.Tensor] = F.relu):
        super().__init__()
        self.hidden_l1 = nn.Linear(input_size, hidden_layer_sizes[0])
        self.hidden_l2 = nn.Linear(hidden_layer_sizes[0], hidden_layer_sizes[1])
        self.hidden_l3 = nn.Linear(hidden_layer_sizes[1], hidden_layer_sizes[2])
        self.hidden_l4 = nn.Linear(hidden_layer_sizes[2], hidden_layer_sizes[3])
        self.output_l5 = nn.Linear(hidden_layer_sizes[3], output_size)
        self.activation_fn = activation_fn
        
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.hidden_l1(inputs)
        x = self.hidden_l2(x)
        x = self.hidden_l3(x)
        x = self.hidden_l4(x)
        x = self.output_l5(x)
        x = self.activation_fn(x)
        return(x)


class alt_three_hidden_layer_MLP(nn.Module): 
    def __init__(self,
                 input_size: int,
                 hidden_layer_sizes: list,
                 output_size: int,
                 activation_fn: Callable[[torch.Tensor], torch.Tensor] = F.relu):
        super().__init__()
        self.hidden_l1 = nn.Linear(input_size, hidden_layer_sizes[0])
        self.hidden_l2 = nn.Linear(hidden_layer_sizes[0], hidden_layer_sizes[1])
        self.hidden_l3 = nn.Linear(hidden_layer_sizes[1], hidden_layer_sizes[2])
        self.output_l4 = nn.Linear(hidden_layer_sizes[2], output_size)
        self.activation_fn = activation_fn
        
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.hidden_l1(inputs)
        x = self.activation_fn(x)
        x = self.hidden_l2(x)
        x = self.activation_fn(x)
        x = self.hidden_l3(x)
        x = self.activation_fn(x)
        x = self.output_l4(x)
        x = self.activation_fn(x)
        return(x)
    
#%% Creating class for converting data from array to tensor
class ToTensor(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, item_id):
        return self.features[item_id, :], self.labels[item_id]
    
#%% Creating a function for computing performance metrics
def get_metrics(labels, predictions):
    predictions_np = predictions.detach().numpy()
    predicted_classes = np.argmax(predictions_np, axis=1)
    
    average_f1_score = f1_score(labels, predicted_classes, average='macro')
    accuracy = accuracy_score(labels, predicted_classes)
     
    return average_f1_score, accuracy

#%% Creating an instance of the network & setting parameters
epochs = 300
eta = 0.005
batch_size = 60
   
input_size = scaled_data_features.shape[1]
hidden_layer_sizes = [100, 100, 100, 100]
output_size = np.unique(data_labels).shape[0]

model = three_hidden_layer_MLP(input_size, hidden_layer_sizes, output_size)

print(f"\n Number of epochs: {epochs}")
print(f"\n Batch size: {batch_size}")
print(f"\n Learning rate: {eta}")
print("\n Model: 3-layer MLP")

#%% Set up the data loading by batch
train_data = ToTensor(scaled_data_features[train_ids, :], data_labels[train_ids])
train_dataloader = DataLoader(train_data, batch_size=batch_size)

test_data = ToTensor(scaled_data_features[test_ids, :], data_labels[test_ids])
test_dataloader = DataLoader(test_data, batch_size=len(test_data))

val_data = ToTensor(scaled_data_features[val_ids, :], data_labels[val_ids])
val_dataloader = DataLoader(val_data, batch_size=len(val_data))

#%% Creating the mini-batch gradient descent function for training the model (Code influenced by labs)
def train_model(model, eta, epochs, train_dataloader, val_dataloader):
    optimizer = SGD(model.parameters(), lr=eta)
    loss_function = nn.CrossEntropyLoss()
    model.train()
    
    best_model_accuracy = 0
    losses = []
    improvement = 999
    prev_loss = 999
    
    for epoch in range (0, epochs):
        if epoch == 0:
            best_model = deepcopy(model)
        
        for batch, (X_train, y_train) in enumerate(train_dataloader):
            train_pred = model.forward(X_train)
            train_loss = loss_function(train_pred, y_train)
            train_average_f1_score, train_accuracy = get_metrics(y_train, train_pred)
            
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
        
        for batch, (X_val, y_val) in enumerate(val_dataloader):
            val_pred = model.forward(X_val)
            val_loss = loss_function(val_pred, y_val)
            val_average_f1_score, val_accuracy = get_metrics(y_val, val_pred)
               
        print("\n Epoch: {} \n Train loss: {:5.5f} \n Train accuracy: {:2.2f}".format(
              epoch+1,
              train_loss.item(),
              train_accuracy)
              +"\n Validation loss: {:5.5f} \n Validation accuracy: {:2.2f}".format(
              val_loss.item(),
              val_accuracy
        ))
        
        improvement = prev_loss - train_loss.item()
        prev_loss = train_loss.item()
        print(f" Reduction in training loss: {improvement}")
        
        if val_accuracy > best_model_accuracy:
            best_model_accuracy = val_accuracy
            best_model = deepcopy(model)
            print("--- Iteration improved performance: new model saved.")
        
        losses.append([train_loss.item(), val_loss.item()])
          
        # Stopping criterion
        if improvement < 0.001:
           print(f"\n Optimisation stopped as stopping criterion met at epoch: {epoch+1}") 
           break
        
    model = best_model
    
    return model, losses

#%% Function to evaluate model performace on test set
def evaluate_model(model, losses, test_dataloader):
    model.eval()
    
    for batch, (X_test, y_test) in enumerate(test_dataloader):
        # Evaluation metrics scores
        test_pred = model.forward(X_test)
        test_average_f1_score, test_accuracy = get_metrics(y_test, test_pred)
        print("\n Neural Network test accuracy: {:2.2f}\n Neural Network test F1 score: {:1.2f} ".format(
            test_average_f1_score, test_accuracy))
              
        test_pred_np = test_pred.detach().numpy()
        test_f1_scores = f1_score(y_test, np.argmax(test_pred_np, axis=1), average=None)
        print(f"\n The Neural Network F1 scores for each of the classes are: {test_f1_scores}")     
               
        nn_precision = precision_recall_fscore_support(y_test, np.argmax(test_pred_np, axis=1))[0]
        print(f"The precision for each class prediction was: {nn_precision}")

        nn_recall= precision_recall_fscore_support(y_test, np.argmax(test_pred_np, axis=1))[1]
        print(f"The recall for each class prediction was: {nn_recall}")

        avg_precision = precision_recall_fscore_support(y_test, np.argmax(test_pred_np, axis=1), average='macro')[0]
        avg_recall= precision_recall_fscore_support(y_test, np.argmax(test_pred_np, axis=1), average='macro')[1]
        print(f"The average precision across classes was: {avg_precision}."
              +f"\n The average recall across classes was: {avg_recall}.")
        
        
        # Confusion matrix plot
        print("\n Confusion matrix:")
        c_mat = confusion_matrix(y_test, np.argmax(test_pred_np, axis=1))
        display = ConfusionMatrixDisplay(c_mat, display_labels=bean_classes)
        display.plot()
        plt.grid(False)
        plt.xticks(rotation=45)
        plt.ylabel("True label", fontname='Times New Roman')
        plt.xlabel("Predicted label", fontname='Times New Roman')
        plt.title("Figure 6: Confusion Matrix: Neural Network", fontname='Times New Roman')
        plt.show()
        
        
        # Loss by epoch plot
        print("\n Loss:")
        fig, ax = plt.subplots()
        losses = np.array(losses)
        ax.plot(losses[:, 0], '-', label='Training Loss', color='grey')
        ax.plot(losses[:, 1], 'o-', label='Validation Loss', color='grey')
        plt.legend(loc='upper right')
        plt.title("Figure 5: Change in Neural Network Loss Across Optimisation Epochs", fontname='Times New Roman')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        
        
        # (Making the predicted and actual labels easier to access, without having to change code above as lazy)
        nn_preds = np.argmax(test_pred_np, axis=1)
        nn_actual = y_test.numpy()
        print(nn_preds)
        print(nn_actual)

        nn_metrics_report = classification_report(nn_actual, nn_preds, target_names=bean_classes, digits=4)
        print("\n The metrics report for the NN is:")
        print(nn_metrics_report)

        # Binarising labals so that ROC/AUC is possible
        nn_preds = label_binarize(nn_preds, classes=[0,1,2,3,4,5,6])
        nn_actual = label_binarize(nn_actual, classes=[0,1,2,3,4,5,6])
        n_classes = 7    
        nn_auc_scores = []
        
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        # Creating list with AUC scores for report
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(nn_actual[:, i], nn_preds[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            nn_auc_scores.append(roc_auc[i])
        
        # Plotting a ROC curve (with AUC score) for each class
        for i in range(n_classes):
            plt.figure()
            plt.plot(fpr[i], tpr[i], label=f"ROC curve (AUC = {roc_auc[i].round(2)})")
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"NN ROC curve for {bean_classes[i]}")
            plt.legend(loc='lower right')
            plt.show()        
        
        # Plotting a ROC curve (with AUC scores) for all classes in one plot
        plt.figure()
        colours = cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])
        for i, colour in zip(range(n_classes), colours):
            plt.plot(fpr[i], tpr[i], color=colour,
                     label=f"{bean_classes[i]} (AUC = {roc_auc[i].round(2)})")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([-0.05, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate", fontname='Times New Roman')
        plt.ylabel("True Positive Rate", fontname='Times New Roman')
        plt.title("Figure 8: ROC curves for NN", fontname='Times New Roman')
        plt.legend(loc='lower right')
        plt.show()

        print(f"The mean AUC score for the NN is: {mean(nn_auc_scores).round(2)}")
        
#%% Run NN
trained_model, losses = train_model(model, eta, epochs, train_dataloader, val_dataloader)

#%% Evaluate NN
evaluate_model(trained_model, losses, test_dataloader)

#%% Build an initial, unoptimised SVM & optimising parameters (with CV)
# Code to build unoptimised SVM (& confusion matrix), so can visualise improvement in performance after optimisation
# clf_svm = SVC(random_state=seed)
# clf_svm.fit(scaled_data_features[train_ids], data_labels[train_ids])

# plot_confusion_matrix(clf_svm,
                      # scaled_data_features[train_ids],
                      # data_labels[train_ids])


# Searching for optimal parameters using cross-validation
svc_params = [{'C': [0.1, 0.5, 1, 10, 100], 
               'gamma': ['scale', 1, 0.1, 0.01, 0.001, 0.0001],
               'kernel': ['rbf']},
              ]

optimal_params = GridSearchCV(
    SVC(),
    svc_params,
    cv=5,
    scoring='accuracy',
    verbose=1
    )

# optimal_params.fit(scaled_data_features[train_ids], data_labels[train_ids])
# print(optimal_params.best_params_)

#%% Alternative parameter optimisation - using validation set rather than CV (Code influenced by labs)
c_options = [0.1, 0.5, 1, 10, 100]
best_c = 0.1
best_c_perf = 0
  
for c in c_options:
    # print("\n Testing c="+str(c)+"...")
    clf_svm = SVC(C=c, kernel='rbf', degree=3, gamma='scale', class_weight=None, random_state=seed)
    clf_svm.fit(scaled_data_features[train_ids], data_labels[train_ids])
    val_pred = clf_svm.predict(scaled_data_features[val_ids])
    
    avg_f1_score = f1_score(data_labels[val_ids], val_pred, average='macro')
    
    if avg_f1_score > best_c_perf:
        best_c = c
        best_c_perf = avg_f1_score
        
print(f"Alternative optimal c for this data is: {best_c}")
   
#%% Final SVM with optimised parameters
clf_svm = SVC(random_state=seed, kernel='rbf', C=100, gamma=0.1, decision_function_shape='ovr')
# vvv Alternative optimised parameters - commented out as parameters in line above perform better.
# clf_svm = SVC(random_state=seed, C=10)
clf_svm.fit(scaled_data_features[train_ids], data_labels[train_ids])


# Evaluation metrics for SVM
test_pred = clf_svm.predict(scaled_data_features[test_ids])

avg_f1_score = f1_score(data_labels[test_ids], test_pred, average='macro')
f1_scores = f1_score(data_labels[test_ids], test_pred, average=None)

print(f"\n The SVM F1 scores for each of the classes are: {f1_scores}")
print(f"\n The average SVM F1 score is: {avg_f1_score}")

acc = accuracy_score(data_labels[test_ids], test_pred)
print(f"\n The overall SVM accuracy is: {acc}")


# Confusion matrix for SVM
c_mat = confusion_matrix(data_labels[test_ids], test_pred)
display = ConfusionMatrixDisplay(c_mat, display_labels=bean_classes)
display.plot()
plt.grid(False)
plt.xticks(rotation=45)
plt.title("Figure 7: Confusion Matrix: Support Vector Machine", fontname='Times New Roman')
plt.ylabel("True label", fontname='Times New Roman')
plt.xlabel("Predicted label", fontname='Times New Roman')
plt.show()


# More evaluation metrics for SVM
y_actual = data_labels[test_ids]

svm_precision = precision_recall_fscore_support(y_actual, test_pred)[0]
print(f"The SVM precision for each class prediction was: {svm_precision}")

svm_recall= precision_recall_fscore_support(y_actual, test_pred)[1]
print(f"The SVM recall for each class prediction was: {svm_recall}")

avg_precision = precision_recall_fscore_support(y_actual, test_pred, average='macro')[0]
avg_recall= precision_recall_fscore_support(y_actual, test_pred, average='macro')[1]
print(f"The average SVM precision across classes was: {avg_precision}."
      +f"\n The average SVM recall across classes was: {avg_recall}.")

svm_metrics_report = classification_report(y_actual, test_pred, target_names=bean_classes)
svm_specificity = recall_score(np.logical_not(y_actual) , np.logical_not(test_pred))

print("\n The metrics report for the SVM is:")
print(svm_metrics_report)
print(f"\n The specificity for the SVM is {svm_specificity}")     

#%%
# class_0 = 0 <-- Checking labels in test dataset are correct order
# for row in range(test_pred.shape[0]):
#     if test_pred[row] == 0:
#         class_0+=1
        
# print(class_0)


# Returning labels to original names (rather than number codes)
labelled_test_pred = test_pred.astype(str)

np.place(labelled_test_pred, labelled_test_pred=='0', 'BARBUNYA')
np.place(labelled_test_pred, labelled_test_pred=='1', 'BOMBAY')
np.place(labelled_test_pred, labelled_test_pred=='2', 'CALI')
np.place(labelled_test_pred, labelled_test_pred=='3', 'DERMASON')
np.place(labelled_test_pred, labelled_test_pred=='4', 'HOROZ')
np.place(labelled_test_pred, labelled_test_pred=='5', 'SEKER')
np.place(labelled_test_pred, labelled_test_pred=='6', 'SIRA')


# Binarising labels for ROC/AUC
y_actual = label_binarize(y_actual, classes=[0,1,2,3,4,5,6])
test_pred = label_binarize(test_pred, classes=[0,1,2,3,4,5,6])
n_classes = 7
svm_auc_scores = []

fpr = dict()
tpr = dict()
roc_auc = dict()

# Creating list with AUC scores for report
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_actual[:, i], test_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    svm_auc_scores.append(roc_auc[i])

# Plotting a ROC curve (with AUC score) for each classes
for i in range(n_classes):
    plt.figure()
    plt.plot(fpr[i], tpr[i], label=f"ROC curve (AUC = {roc_auc[i].round(2)})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"SVM ROC curve for {bean_classes[i]}")
    plt.legend(loc='lower right')
    plt.show()

# Plotting a ROC curve (with AUC scores) for all classes in one plot
plt.figure()
colours = cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])
for i, colour in zip(range(n_classes), colours):
    plt.plot(fpr[i], tpr[i], color=colour,
             label=f"{bean_classes[i]} (AUC = {roc_auc[i].round(2)})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate", fontname='Times New Roman')
plt.ylabel("True Positive Rate", fontname='Times New Roman')
plt.title("Figure 9: ROC curves for SVM", fontname='Times New Roman')
plt.legend(loc='lower right')
plt.show()

print(mean(svm_auc_scores).round(2))

# #%% Writing output from SVM (the best-performing model) to a CSV file
# with open('svm_output.csv', 'w', newline='') as f:
#     writer = csv.writer(f)
#     for row in range(labelled_test_pred.shape[0]):
#         writer.writerow([test_ids[row], labelled_test_pred[row]])







