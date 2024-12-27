# Import necessary libraries
import numpy as np
import pandas as pd
import csv
import os
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score


# Load training data
train_metadata_path = 'modified2.csv'
test_metadata_path = 'test-metadata4.csv'
train_metadata = pd.read_csv(train_metadata_path)
test_metadata = pd.read_csv(test_metadata_path)

# Remove unnecessary columns
columns_to_remove = ['lesion_id', 'iddx_2', 'iddx_3', 'iddx_4', 'iddx_5', 'mel_mitotic_index', 'mel_thick_mm']
train_metadata_cleaned = train_metadata.drop(columns=columns_to_remove, errors='ignore')
train_metadata_cleaned_no_nulls = train_metadata_cleaned.dropna()

# Check the class distribution before balancing
print("Class distribution before balancing:")
print(train_metadata_cleaned_no_nulls['target'].value_counts())

# Separate the majority (benign - class 0) and minority classes (melanoma - class 2, basal cell carcinoma - class 1, squamous cell carcinoma - class 3)
majority_class = train_metadata_cleaned_no_nulls[train_metadata_cleaned_no_nulls['target'] == 0]
minority_classes = train_metadata_cleaned_no_nulls[train_metadata_cleaned_no_nulls['target'].isin([1, 2, 3])]

# Upsample minority classes to match the majority class size
minority_classes_upsampled = resample(minority_classes, replace=True, n_samples=len(majority_class), random_state=42)

# Combine majority class with upsampled minority classes
train_metadata_balanced = pd.concat([majority_class, minority_classes_upsampled])

# Display the shape and class distribution of the balanced DataFrame
print("Shape after balancing classes:", train_metadata_balanced.shape)
print("Class Distribution after balancing:\n", train_metadata_balanced['target'].value_counts())

# Encode 'sex' and other non-numeric columns
train_metadata_balanced['sex'] = train_metadata_balanced['sex'].map({'male': 1, 'female': 0})
anatom_site_mapping = {'posterior torso': 1, 'lower extremity': 2, 'anterior torso': 3, 'upper extremity': 4, 'head/neck': 5}
train_metadata_balanced['anatom_site_general'] = train_metadata_balanced['anatom_site_general'].apply(lambda x: anatom_site_mapping.get(x, 0))

# Encode 'tbp_lv_location' and 'tbp_lv_location_simple' columns
tbp_lv_location_mapping = {'Torso Front Top Half': 1, 'Torso Back Top Third': 2, 'Head & Neck': 3, 'Torso Back Middle Third': 4, 'Left Leg - Lower': 5,
                           'Right Leg - Lower': 6, 'Torso Front Bottom Half': 7, 'Left Arm - Upper': 8, 'Left Leg - Upper': 9, 'Right Arm - Upper': 10,
                           'Right Leg - Upper': 11, 'Left Arm - Lower': 12, 'Right Arm - Lower': 13, 'Torso Back Bottom Third': 14, 'Left Leg': 15,
                           'Right Leg': 16, 'Left Arm': 17, 'Right Arm': 18}
train_metadata_balanced['tbp_lv_location'] = train_metadata_balanced['tbp_lv_location'].apply(lambda x: tbp_lv_location_mapping.get(x, 0))

tbp_lv_location_simple_mapping = {'Torso Back': 1, 'Torso Front': 2, 'Left Leg': 3, 'Head & Neck': 4, 'Right Leg': 5, 'Left Arm': 6, 'Right Arm': 7}
train_metadata_balanced['tbp_lv_location_simple'] = train_metadata_balanced['tbp_lv_location_simple'].apply(lambda x: tbp_lv_location_simple_mapping.get(x, 0))

# Split features and target
X = train_metadata_balanced.drop(columns=['isic_id', 'target', 'patient_id', 'image_type', 'tbp_tile_type', 'attribution', 'copyright_license', 'iddx_full', 'iddx_1'])
y = train_metadata_balanced['target']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y.values, dtype=torch.long)  # Multiclass classification requires long dtype

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

# Create DataLoaders
batch_size = 64
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Define a simple neural network for multiclass classification
class SimpleNN(nn.Module):
    def __init__(self, input_dim, num_classes):
       
       super(SimpleNN, self).__init__()
    #    self.fc1 = nn.Linear(input_dim, 128)  # Increase neurons
     #   self.fc2 = nn.Linear(128, 64)
     #   self.fc3 = nn.Linear(64, 32)
     #   self.fc4 = nn.Linear(32, num_classes)
      #  self.softmax = nn.Softmax(dim=1) 
    
  #  def forward(self, x):
  #      x = torch.relu(self.fc1(x))
  #      x = torch.relu(self.fc2(x))
  #      x = torch.relu(self.fc3(x))
  #      x = self.fc4(x)
  #      return self.softmax(x)

       self.fc1 = nn.Linear(input_dim, 64)
       self.fc2 = nn.Linear(64, 32)
       self.fc3 = nn.Linear(32, num_classes)  # Final layer for multiclass classification
       self.softmax = nn.Softmax(dim=1)  # Softmax for multiclass classification

    def forward(self, x):
         x = torch.relu(self.fc1(x))
         x = torch.relu(self.fc2(x))
         x = self.fc3(x)
         x = self.softmax(x)
         return x

# Model initialization
input_dim = X_train.shape[1]
num_classes = 4  # 4 classes: benign, melanoma, basal cell carcinoma, squamous cell carcinoma
model = SimpleNN(input_dim, num_classes)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()  # CrossEntropyLoss for multiclass classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Track metrics during training
train_accuracies = []
val_accuracies = []
f1_scores = []
#Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import random
random.seed(42)
# Train the model
num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    correct_train = 0
    total_train = 0
    
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)
        correct_train += (predicted == y_batch).sum().item()
        total_train += y_batch.size(0)
    
    train_accuracy = correct_train / total_train
    train_accuracies.append(train_accuracy)
    
    # Validation phase
    model.eval()
    val_loss = 0
    correct_val = 0
    total_val = 0
    all_val_preds = []
    all_val_labels = []

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_val += (predicted == y_batch).sum().item()
            total_val += y_batch.size(0)

            all_val_preds.extend(predicted.numpy())
            all_val_labels.extend(y_batch.numpy())

    val_accuracy = correct_val / total_val
    val_accuracies.append(val_accuracy)
    f1 = f1_score(all_val_labels, all_val_preds, average='weighted')
    f1_scores.append(f1)
    
    print(f'Epoch {epoch+1}/{num_epochs}, Train Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}, F1 Score: {f1:.4f}')

class_labels = {0: 'Benign', 1: 'Basal cell carcinoma', 2: 'Melanoma', 3: 'Squamous cell carcinoma'}
# Plot epoch accuracy curve
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_accuracies, label='Training Accuracy', marker='o')
plt.plot(range(1, num_epochs + 1), val_accuracies, label='Validation Accuracy', marker='o')
plt.title('Epoch Accuracy Curve')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Plot F1 score curve
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), f1_scores, label='F1 Score', marker='o', color='green')
plt.title('Epoch F1 Score Curve')
plt.xlabel('Epochs')
plt.ylabel('F1 Score')
plt.legend()
plt.grid(True)
plt.show()

# ROC AUC curve
y_val_pred_prob = model(X_val).detach().numpy()
y_true_binarized = label_binarize(y_val.numpy(), classes=[0, 1, 2, 3])
n_classes = y_true_binarized.shape[1]

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], y_val_pred_prob[:, i])
    roc_auc[i] = roc_auc_score(y_true_binarized[:, i], y_val_pred_prob[:, i])

plt.figure(figsize=(10, 7))
for i, color in zip(range(n_classes), ['blue', 'red', 'green', 'orange']):
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'{class_labels[i]} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC AUC Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# Precision-Recall Curve
precision = dict()
recall = dict()
average_precision = dict()

for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(y_true_binarized[:, i], y_val_pred_prob[:, i])
    average_precision[i] = average_precision_score(y_true_binarized[:, i], y_val_pred_prob[:, i])

plt.figure(figsize=(10, 7))
for i, color in zip(range(n_classes), ['blue', 'red', 'green', 'orange']):
    plt.plot(recall[i], precision[i], color=color, lw=2, label=f'{class_labels[i]} (AP = {average_precision[i]:.2f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
plt.grid(True)
plt.show()





final_accuracy = val_accuracies[-1]  # Last epoch's validation accuracy

# Convert predictions and true labels to numpy arrays for metric calculation
all_val_preds = np.array(all_val_preds)
all_val_labels = np.array(all_val_labels)

# Calculate precision, recall, and F1 scores for each class
precision = precision_score(all_val_labels, all_val_preds, average=None)
recall = recall_score(all_val_labels, all_val_preds, average=None)
f1 = f1_score(all_val_labels, all_val_preds, average=None)

# Calculate weighted average for each metric
average_precision_score_weighted = precision_score(all_val_labels, all_val_preds, average='weighted')
average_recall_score_weighted = recall_score(all_val_labels, all_val_preds, average='weighted')
average_f1_score_weighted = f1_score(all_val_labels, all_val_preds, average='weighted')

# Calculate ROC AUC scores for each class
y_val_pred_prob = model(X_val).detach().numpy()
roc_auc = []
for i in range(n_classes):
    roc_auc.append(roc_auc_score(y_true_binarized[:, i], y_val_pred_prob[:, i]))

# Print final average metrics
print("\nFinal Model Performance on Validation Set:")
print(f"Average Validation Accuracy: {final_accuracy:.4f}")
print(f"Precision (per class): {precision}")
print(f"Recall (per class): {recall}")
print(f"F1 Score (per class): {f1}")
print(f"Weighted Average Precision: {average_precision_score_weighted:.4f}")
print(f"Weighted Average Recall: {average_recall_score_weighted:.4f}")
print(f"Weighted Average F1 Score: {average_f1_score_weighted:.4f}")
print(f"ROC AUC (per class): {roc_auc}")
print(f"Average ROC AUC: {np.mean(roc_auc):.4f}")

# Test the model with new test data
test_metadata_cleaned = test_metadata.drop(columns=columns_to_remove, errors='ignore')

# Ensure the test data has the same columns as the training data
for col in X.columns:
    if col not in test_metadata_cleaned.columns:
        test_metadata_cleaned[col] = 0  # Add missing columns with default value (0)

test_metadata_cleaned = test_metadata_cleaned.loc[:, X.columns]  # Align columns
test_metadata_cleaned['sex'] = test_metadata_cleaned['sex'].map({'male': 1, 'female': 0})
test_metadata_cleaned['anatom_site_general'] = test_metadata_cleaned['anatom_site_general'].apply(lambda x: anatom_site_mapping.get(x, 0))
test_metadata_cleaned['tbp_lv_location'] = test_metadata_cleaned['tbp_lv_location'].apply(lambda x: tbp_lv_location_mapping.get(x, 0))
test_metadata_cleaned['tbp_lv_location_simple'] = test_metadata_cleaned['tbp_lv_location_simple'].apply(lambda x: tbp_lv_location_simple_mapping.get(x, 0))

X_test = test_metadata_cleaned.drop(columns=['isic_id', 'patient_id', 'image_type', 'tbp_tile_type', 'attribution', 'copyright_license'], errors='ignore')
X_test_scaled = scaler.transform(X_test)  # Scale test data with the same scaler

# Convert to tensor
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

# Make predictions
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    _, test_predictions = torch.max(test_outputs, 1)


