import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import GRU, Dense, Dropout, Masking, Conv1D, GlobalAveragePooling1D, Concatenate, Input, BatchNormalization, SpatialDropout1D, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns

# Paths to the dataset and output directory
data_dir = r"path for keypoints dataset"
output_dir = r"output directory of each folds"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Parameters
SEQUENCE_LENGTH = 60
NUM_KEYPOINTS = 126  # Assuming x, y, z for each keypoint (42 points * 3)
NUM_CLASSES = 10    # Based on the number of classes in the dataset
BATCH_SIZE = 16
EPOCHS = 50
N_SPLITS = 5  # Number of splits for K-fold cross-validation

# Function to load data from directories
def load_data(data_dir):
    X, y = [], []
    labels = os.listdir(data_dir)
    label_map = {label: idx for idx, label in enumerate(labels)}
    
    for label in labels:
        label_path = os.path.join(data_dir, label)
        if os.path.isdir(label_path):
            for subfolder in os.listdir(label_path):
                subfolder_path = os.path.join(label_path, subfolder)
                
                # Load keypoints
                original_file_path = os.path.join(subfolder_path, 'keypoints.npy')
                if os.path.exists(original_file_path):
                    keypoints = np.load(original_file_path)
                    X.append(keypoints)
                    y.append(label_map[label])

    X = np.array(X)
    y = np.array(y)
    
    # Replace NaN values with zeros to handle missing keypoints
    X = np.nan_to_num(X, nan=0.0)
    return X, y, labels

# Load datasets
print("Loading dataset...")
X, y, labels = load_data(data_dir)

# Save labels to a file
with open(os.path.join(output_dir, 'calendar_labels.txt'), 'w') as f:
    for label in labels:
        f.write(f"{label}\n")

# Define K-fold cross-validator
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
fold_no = 1
all_fold_accuracies = []
all_histories = []
all_y_true, all_y_pred = [], []
models = []  # List to store each trained model

# Start K-fold cross-validation
for train_index, val_index in kf.split(X):
    print(f"Starting Fold {fold_no}...")
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    # Define the model with reduced complexity and additional regularization
    input_layer = Input(shape=(SEQUENCE_LENGTH, NUM_KEYPOINTS))

    # Feature extraction with Conv1D layers
    combined_features = Conv1D(filters=128, kernel_size=3, activation=None, padding='same', kernel_regularizer=l2(0.01))(input_layer)
    combined_features = BatchNormalization()(combined_features)
    combined_features = LeakyReLU(alpha=0.1)(combined_features)
    combined_features = SpatialDropout1D(0.4)(combined_features)
    combined_features = Conv1D(filters=64, kernel_size=3, activation=None, padding='same', kernel_regularizer=l2(0.01))(combined_features)
    combined_features = BatchNormalization()(combined_features)
    combined_features = LeakyReLU(alpha=0.1)(combined_features)
    combined_features = GlobalAveragePooling1D()(combined_features)

    # Temporal feature extraction with GRU layers
    temporal_features = Masking(mask_value=0.0)(input_layer)
    temporal_features = GRU(128, return_sequences=True, kernel_regularizer=l2(0.01), dropout=0.4, recurrent_dropout=0.3)(temporal_features)
    temporal_features = BatchNormalization()(temporal_features)
    temporal_features = GRU(64, return_sequences=False, kernel_regularizer=l2(0.01), dropout=0.4, recurrent_dropout=0.3)(temporal_features)
    temporal_features = BatchNormalization()(temporal_features)

    # Concatenate spatial and temporal features
    concatenated = Concatenate()([combined_features, temporal_features])

    # Fully connected layers for classification
    dense = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(concatenated)
    dropout = Dropout(0.6)(dense)
    output_layer = Dense(NUM_CLASSES, activation='softmax')(dropout)

    # Define and compile the model
    model = Model(inputs=input_layer, outputs=output_layer)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model for the current fold
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)
        ]
    )

    # Save model for ensemble
    model_path = os.path.join(output_dir, f'model_fold_{fold_no}.h5')
    model.save(model_path)
    models.append(model)

    # Save history for averaging later
    all_histories.append(history)

    # Evaluate the model on the validation set
    val_loss, val_acc = model.evaluate(X_val, y_val)
    all_fold_accuracies.append(val_acc)
    print(f"Fold {fold_no} Validation Accuracy: {val_acc:.2f}")

    # Predict and collect predictions and true labels for the confusion matrix
    y_pred = np.argmax(model.predict(X_val), axis=1)
    all_y_true.extend(y_val)
    all_y_pred.extend(y_pred)

    # Paths for fold-specific outputs
    fold_output_dir = os.path.join(output_dir, f'FOLD {fold_no}')
    os.makedirs(fold_output_dir, exist_ok=True)

    # Plot and save the training history for the current fold
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy / Loss')
    plt.legend()
    plt.title(f'Training and Validation Accuracy & Loss for Fold {fold_no}')
    plt.savefig(os.path.join(fold_output_dir, 'training_history.png'))
    plt.close()

    # Generate and save the confusion matrix for the current fold
    conf_matrix = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix for Fold {fold_no}')
    plt.savefig(os.path.join(fold_output_dir, 'confusion_matrix.png'))
    plt.close()

    # Generate and save the classification report for the current fold
    class_report = classification_report(y_val, y_pred, target_names=labels)
    with open(os.path.join(fold_output_dir, 'classification_report.txt'), 'w') as f:
        f.write(class_report)

    print(f"Confusion matrix and classification report saved for Fold {fold_no}.")
    fold_no += 1

# Ensemble prediction
print("Generating ensemble predictions...")
def ensemble_predict(models, X):
    predictions = [model.predict(X) for model in models]
    avg_predictions = np.mean(predictions, axis=0)
    return np.argmax(avg_predictions, axis=1)

ensemble_preds = ensemble_predict(models, X)

# Generate and save the confusion matrix for ensemble
conf_matrix_ensemble = confusion_matrix(y, ensemble_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix_ensemble, annot=True, cmap='Blues', fmt='g', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix for Ensemble Model')
plt.savefig(os.path.join(output_dir, 'confusion_matrix_ensemble.png'))
plt.close()

# Generate and save the classification report for the ensemble
class_report_ensemble = classification_report(y, ensemble_preds, target_names=labels)
with open(os.path.join(output_dir, 'classification_report_ensemble.txt'), 'w') as f:
    f.write(class_report_ensemble)

print("Ensemble model evaluation completed.")
print(f"Average K-Fold Validation Accuracy: {np.mean(all_fold_accuracies):.2f}")
