import numpy as np
import scipy.signal as sg
import wfdb
import os
import csv
import gc
import pickle
import matplotlib.pyplot as plt
import scipy.stats
import time
import sklearn
import operator
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, make_scorer
from pathlib import Path
from sklearn import decomposition
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv1D, MaxPooling1D, Input, Concatenate, BatchNormalization
from tensorflow.keras.callbacks import History
from tensorflow.keras.utils import to_categorical
import seaborn as sns

dataset = [
   '101', '106', '108', '109', '112', '114', '115', '116', '118', '119', '122', '124', '201', '203', '205', '207',
   '208', '209', '215', '220', '223', '230', '233', '100', '103', '105', '111', '113', '117', '121', '123', '200',
   '202', '210', '212', '213', '214', '219', '221', '222', '228', '231', '232', '234'
]

## Preprocessing
sampling_rate = 360
winL = 90
winR = 90
size_RR_max = 20

def filter(signal):
    baseline = sg.medfilt(sg.medfilt(signal, int(0.2 * sampling_rate) - 1), int(0.6 * sampling_rate) - 1)
    filtered_signal = signal - baseline
    return filtered_signal

MITBIH_classes = ['N', 'L', 'R', 'e', 'j', 'A', 'a', 'J', 'S', 'V', 'E', 'F', 'P', '/', 'f', 'u', 'Q']
AAMI_classes = []
AAMI_classes.append(['N', 'L', 'R'])                    # N
AAMI_classes.append(['A', 'a', 'J', 'S', 'e', 'j'])     # SVEB
AAMI_classes.append(['V', 'E'])                         # VEB
AAMI_classes.append(['F'])                              # F
AAMI_classes.append(['P', '/', 'f', 'u', 'Q'])              # Q

def extract_morphology_features(beat):
    """Extract comprehensive morphology features from ECG beat"""
    try:
        beat = np.array(beat)
        
        # 1. Statistical Features
        mean_val = np.mean(beat)
        std_val = np.std(beat)
        var_val = np.var(beat)
        skew_val = skew(beat)
        kurt_val = kurtosis(beat)
        max_val = np.max(beat)
        min_val = np.min(beat)
        peak_to_peak = max_val - min_val
        
        # 2. Peak Detection Features
        peaks, _ = find_peaks(beat, height=np.mean(beat) + 0.1*np.std(beat), distance=10)
        num_peaks = len(peaks)
        
        # R-peak features (highest peak)
        r_peak_idx = np.argmax(beat)
        r_peak_amplitude = beat[r_peak_idx]
        r_peak_position = r_peak_idx / len(beat)  # Normalized position
        
        # 3. QRS Complex Features
        # Define QRS region around R-peak
        qrs_start = max(0, r_peak_idx - 20)
        qrs_end = min(len(beat), r_peak_idx + 20)
        qrs_duration = qrs_end - qrs_start
        qrs_width = qrs_duration / sampling_rate * 1000  # in milliseconds
        qrs_area = np.trapz(np.abs(beat[qrs_start:qrs_end]))
        
        # 4. P and T wave features
        # P wave (before QRS) - first third of signal
        p_wave_region = beat[:len(beat)//3]
        p_wave_max = np.max(p_wave_region)
        p_wave_min = np.min(p_wave_region)
        p_wave_area = np.trapz(np.abs(p_wave_region))
        
        # T wave (after QRS) - last third of signal
        t_wave_region = beat[2*len(beat)//3:]
        t_wave_max = np.max(t_wave_region)
        t_wave_min = np.min(t_wave_region)
        t_wave_area = np.trapz(np.abs(t_wave_region))
        
        # 5. Slope and Gradient Features
        beat_diff = np.diff(beat)
        max_positive_slope = np.max(beat_diff) if len(beat_diff) > 0 else 0
        max_negative_slope = np.min(beat_diff) if len(beat_diff) > 0 else 0
        mean_slope = np.mean(beat_diff) if len(beat_diff) > 0 else 0
        slope_variance = np.var(beat_diff) if len(beat_diff) > 0 else 0
        
        # 6. Interval Features
        # Pre and post R-peak intervals
        pre_r_interval = r_peak_idx
        post_r_interval = len(beat) - r_peak_idx
        
        # 7. Energy and Power Features
        signal_energy = np.sum(beat**2)
        normalized_energy = signal_energy / len(beat)
        rms = np.sqrt(np.mean(beat**2))
        
        # 8. Zero Crossing Features
        zero_crossings = np.sum(np.diff(np.signbit(beat)))
        
        # 9. Amplitude Ratios
        qrs_p_ratio = r_peak_amplitude / (p_wave_max + 1e-8)  # Avoid division by zero
        qrs_t_ratio = r_peak_amplitude / (t_wave_max + 1e-8)
        
        # 10. Heart Rate Variability Approximation
        # Using beat morphology as proxy
        beat_variability = np.std(beat) / (np.mean(np.abs(beat)) + 1e-8)
        
        # 11. Waveform Symmetry
        beat_center = len(beat) // 2
        left_half = beat[:beat_center]
        right_half = beat[beat_center:]
        
        # Ensure equal length for comparison
        min_len = min(len(left_half), len(right_half))
        left_half = left_half[:min_len]
        right_half = right_half[:min_len]
        
        symmetry_correlation = np.corrcoef(left_half, right_half[::-1])[0, 1] if min_len > 1 else 0
        
        # 12. Temporal Features
        # Time to peak from start
        time_to_peak = r_peak_idx / len(beat)
        
        # Upstroke and downstroke characteristics
        upstroke_area = np.trapz(beat[:r_peak_idx]) if r_peak_idx > 0 else 0
        downstroke_area = np.trapz(beat[r_peak_idx:]) if r_peak_idx < len(beat) else 0
        
        # Combine all features
        morphology_features = [
            # Statistical (8 features)
            mean_val, std_val, var_val, skew_val, kurt_val, 
            max_val, min_val, peak_to_peak,
            
            # Peak features (4 features)
            num_peaks, r_peak_amplitude, r_peak_position, qrs_duration,
            
            # QRS features (2 features)
            qrs_width, qrs_area,
            
            # P and T wave features (6 features)
            p_wave_max, p_wave_min, p_wave_area,
            t_wave_max, t_wave_min, t_wave_area,
            
            # Slope features (4 features)
            max_positive_slope, max_negative_slope, mean_slope, slope_variance,
            
            # Interval features (2 features)
            pre_r_interval, post_r_interval,
            
            # Energy features (3 features)
            signal_energy, normalized_energy, rms,
            
            # Other features (8 features)
            zero_crossings, qrs_p_ratio, qrs_t_ratio, beat_variability,
            symmetry_correlation, time_to_peak, upstroke_area, downstroke_area
        ]
        
        return np.array(morphology_features, dtype=np.float32)
        
    except Exception as e:
        print(f"Error extracting morphology features: {e}")
        # Return zero features if extraction fails
        return np.zeros(37, dtype=np.float32)  # Total expected features

def load(record, symbol, sample, winL, winR):
    beat = []
    class_ID = []

    for a in range(len(sample)):
        pos = sample[a]
        classAnttd = symbol[a]
        if (pos > size_RR_max) and (pos < (len(record) - size_RR_max)):
            index, value = max(enumerate(record[pos - size_RR_max : pos + size_RR_max]), key=operator.itemgetter(1))
            pos = (pos - size_RR_max) + index

        if classAnttd in MITBIH_classes:
            if(pos > winL and pos < (len(record) - winR)):
                beat.append((record[pos - winL : pos + winR]))
                for i in range(0, len(AAMI_classes)):
                    if classAnttd in AAMI_classes[i]:
                        class_AAMI = i
                        break #exit loop
                class_ID.append(class_AAMI)

    return pos, beat, class_ID

# Extract beats and classes
beats, classes = [], []
path = './dataset/'

print("Loading ECG data...")
for i in dataset:
    x = path + i
    try:
        sig = wfdb.rdrecord(x)
        annotation = wfdb.rdann(x, "atr")

        filtered_signal = filter(sig.p_signal[:, 0])
        pos, beat, class_ID = load(filtered_signal, annotation.symbol, annotation.sample, winL, winR)

        beats.extend(beat)
        classes.extend(class_ID)
        print(f"Processed {i}: {len(beat)} beats")
    except Exception as e:
        print(f"Error processing {i}: {e}")

print(f"Total beats extracted: {len(beats)}")
print(f"Beat length: {len(beats[0]) if beats else 0}")

# Extract morphology features
print("Extracting morphology features...")
morphology_features = []
processed_beats = []

for i, beat in enumerate(beats):
    try:
        # Extract morphology features
        morph_features = extract_morphology_features(beat)
        morphology_features.append(morph_features)
        processed_beats.append(beat)
        
        if i % 10000 == 0:
            print(f"Processed {i} beats for morphology features")
            
    except Exception as e:
        print(f"Error processing beat {i}: {e}")
        continue

# Align classes with processed data
aligned_classes = classes[:len(morphology_features)]

print(f"Morphology features shape: {np.array(morphology_features).shape}")
print(f"Raw beats shape: {np.array(processed_beats).shape}")
print(f"Number of classes: {len(aligned_classes)}")

# Split data
print("Splitting data...")
indices = list(range(len(morphology_features)))
train_idx, test_idx = train_test_split(indices, test_size=0.30, random_state=42, stratify=aligned_classes)
train_idx, val_idx = train_test_split(train_idx, test_size=0.143, random_state=42, 
                                     stratify=[aligned_classes[i] for i in train_idx])

# Split morphology features
morph_features_array = np.array(morphology_features)
x_train_morph = morph_features_array[train_idx]
x_val_morph = morph_features_array[val_idx]
x_test_morph = morph_features_array[test_idx]

# Split raw beats for CNN
raw_beats_array = np.array(processed_beats)
x_train_raw = raw_beats_array[train_idx]
x_val_raw = raw_beats_array[val_idx]
x_test_raw = raw_beats_array[test_idx]

# Split classes
y_train = [aligned_classes[i] for i in train_idx]
y_val = [aligned_classes[i] for i in val_idx]
y_test = [aligned_classes[i] for i in test_idx]

print(f"Training set: {len(x_train_morph)} samples")
print(f"Validation set: {len(x_val_morph)} samples")
print(f"Test set: {len(x_test_morph)} samples")

# Normalize morphology features
print("Normalizing morphology features...")
scaler = StandardScaler()
x_train_morph_scaled = scaler.fit_transform(x_train_morph)
x_val_morph_scaled = scaler.transform(x_val_morph)
x_test_morph_scaled = scaler.transform(x_test_morph)

# Reshape raw beats for CNN (add channel dimension)
x_train_raw = x_train_raw.reshape(x_train_raw.shape[0], x_train_raw.shape[1], 1)
x_val_raw = x_val_raw.reshape(x_val_raw.shape[0], x_val_raw.shape[1], 1)
x_test_raw = x_test_raw.reshape(x_test_raw.shape[0], x_test_raw.shape[1], 1)

print(f"Final shapes:")
print(f"Morphology features: {x_train_morph_scaled.shape}")
print(f"Raw beats: {x_train_raw.shape}")

# Save processed data
print("Saving processed data...")
with open('./dataset/morphology_data.pkl', "wb") as f:
    pickle.dump({
        'x_train_morph': x_train_morph_scaled,
        'x_val_morph': x_val_morph_scaled,
        'x_test_morph': x_test_morph_scaled,
        'x_train_raw': x_train_raw,
        'x_val_raw': x_val_raw,
        'x_test_raw': x_test_raw,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'scaler': scaler
    }, f, protocol=4)

# Prepare data for training
print("Preparing data for training...")

# One hot encode y
y_train_cat = to_categorical(y_train, num_classes=5)
y_val_cat = to_categorical(y_val, num_classes=5)
y_test_cat = to_categorical(y_test, num_classes=5)

# Create hybrid model combining morphology features and CNN
def create_morphology_cnn_model(morph_input_shape, raw_input_shape, num_classes=5):
    """
    Hybrid model combining morphology features and CNN on raw signals
    """
    # Morphology branch (handcrafted features)
    morph_input = Input(shape=morph_input_shape, name='morphology_input')
    morph_x = Dense(128, activation='relu', name='morph_dense1')(morph_input)
    morph_x = BatchNormalization(name='morph_bn1')(morph_x)
    morph_x = Dropout(0.3, name='morph_dropout1')(morph_x)
    morph_x = Dense(64, activation='relu', name='morph_dense2')(morph_x)
    morph_x = BatchNormalization(name='morph_bn2')(morph_x)
    morph_x = Dropout(0.2, name='morph_dropout2')(morph_x)
    morph_features = Dense(32, activation='relu', name='morph_features')(morph_x)
    
    # CNN branch (raw signal)
    raw_input = Input(shape=raw_input_shape, name='raw_input')
    
    # First CNN block
    cnn_x = Conv1D(64, kernel_size=7, activation="relu", name='conv1')(raw_input)
    cnn_x = BatchNormalization(name='cnn_bn1')(cnn_x)
    cnn_x = Conv1D(64, kernel_size=5, activation="relu", name='conv2')(cnn_x)
    cnn_x = BatchNormalization(name='cnn_bn2')(cnn_x)
    cnn_x = MaxPooling1D(pool_size=2, name='pool1')(cnn_x)
    cnn_x = Dropout(0.25, name='cnn_dropout1')(cnn_x)
    
    # Second CNN block
    cnn_x = Conv1D(128, kernel_size=5, activation="relu", name='conv3')(cnn_x)
    cnn_x = BatchNormalization(name='cnn_bn3')(cnn_x)
    cnn_x = Conv1D(128, kernel_size=3, activation="relu", name='conv4')(cnn_x)
    cnn_x = BatchNormalization(name='cnn_bn4')(cnn_x)
    cnn_x = MaxPooling1D(pool_size=2, name='pool2')(cnn_x)
    cnn_x = Dropout(0.3, name='cnn_dropout2')(cnn_x)
    
    # Third CNN block
    cnn_x = Conv1D(256, kernel_size=3, activation="relu", name='conv5')(cnn_x)
    cnn_x = BatchNormalization(name='cnn_bn5')(cnn_x)
    cnn_x = MaxPooling1D(pool_size=2, name='pool3')(cnn_x)
    cnn_x = Dropout(0.4, name='cnn_dropout3')(cnn_x)
    
    # Global pooling instead of flatten to reduce parameters
    cnn_x = tf.keras.layers.GlobalAveragePooling1D(name='global_pool')(cnn_x)
    cnn_features = Dense(64, activation='relu', name='cnn_features')(cnn_x)
    
    # Combine branches
    combined = Concatenate(name='combine_features')([morph_features, cnn_features])
    
    # Classification layers
    x = Dense(128, activation='relu', name='combined_dense1')(combined)
    x = BatchNormalization(name='combined_bn1')(x)
    x = Dropout(0.5, name='combined_dropout1')(x)
    
    x = Dense(64, activation='relu', name='combined_dense2')(x)
    x = BatchNormalization(name='combined_bn2')(x)
    x = Dropout(0.3, name='combined_dropout2')(x)
    
    outputs = Dense(num_classes, activation='softmax', name='predictions')(x)
    
    model = Model(inputs=[morph_input, raw_input], outputs=outputs, 
                  name='morphology_cnn_hybrid')
    
    return model

# Create model
print("Creating hybrid Morphology + CNN model...")
model = create_morphology_cnn_model(
    morph_input_shape=(x_train_morph_scaled.shape[1],),
    raw_input_shape=x_train_raw.shape[1:]
)

# Compile model
model.compile(
    loss='categorical_crossentropy', 
    optimizer='adam', 
    metrics=['accuracy', 'precision', 'recall']
)

model.summary()
print(f"Total parameters: {model.count_params():,}")

# Train model
print("Training hybrid model...")
history_result = model.fit(
    [x_train_morph_scaled, x_train_raw], y_train_cat,
    batch_size=64,
    epochs=10,
    verbose=1,
    validation_data=([x_val_morph_scaled, x_val_raw], y_val_cat),
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, monitor='val_accuracy'),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-7),
        tf.keras.callbacks.ModelCheckpoint('./model/best_morphology_cnn.h5', 
                                         save_best_only=True, monitor='val_accuracy')
    ]
)

# Evaluate model
print("\nModel Evaluation:")
train_score = model.evaluate([x_train_morph_scaled, x_train_raw], y_train_cat, verbose=0)
val_score = model.evaluate([x_val_morph_scaled, x_val_raw], y_val_cat, verbose=0)
test_score = model.evaluate([x_test_morph_scaled, x_test_raw], y_test_cat, verbose=0)

print('Train - Loss: {:.4f}, Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}'.format(
    train_score[0], train_score[1], train_score[2], train_score[3]))
print('Validation - Loss: {:.4f}, Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}'.format(
    val_score[0], val_score[1], val_score[2], val_score[3]))
print('Test - Loss: {:.4f}, Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}'.format(
    test_score[0], test_score[1], test_score[2], test_score[3]))

# Predictions and evaluation
y_pred = model.predict([x_test_morph_scaled, x_test_raw])
y_pred_label = np.argmax(y_pred, axis=1)
y_true_label = np.argmax(y_test_cat, axis=1)

# Classification results
cm = confusion_matrix(y_true_label, y_pred_label)
print("\nConfusion Matrix:")
print(cm)

class_names = ['Normal', 'SVEB', 'VEB', 'Fusion', 'Unknown']
report = classification_report(y_true_label, y_pred_label, 
                             target_names=class_names, digits=4, output_dict=True)
print("\nClassification Report:")
print(classification_report(y_true_label, y_pred_label, 
                          target_names=class_names, digits=4))

# Overall metrics
print(f"\nOverall Metrics:")
print(f"Macro F1-Score: {f1_score(y_true_label, y_pred_label, average='macro'):.4f}")
print(f"Weighted F1-Score: {f1_score(y_true_label, y_pred_label, average='weighted'):.4f}")

# Enhanced visualization
plt.figure(figsize=(20, 12))

# Training history
plt.subplot(3, 4, 1)
plt.plot(history_result.history['accuracy'], label='Training Accuracy', linewidth=2)
plt.plot(history_result.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
plt.title('Morphology + CNN Model Accuracy', fontsize=12, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(3, 4, 2)
plt.plot(history_result.history['loss'], label='Training Loss', linewidth=2)
plt.plot(history_result.history['val_loss'], label='Validation Loss', linewidth=2)
plt.title('Morphology + CNN Model Loss', fontsize=12, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

# Confusion Matrix
plt.subplot(3, 4, 3)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix', fontsize=12, fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

# Per-class F1 scores
plt.subplot(3, 4, 4)
f1_scores = [report[name.lower()]['f1-score'] for name in class_names if name.lower() in report]
bars = plt.bar(range(len(f1_scores)), f1_scores, color=['blue', 'green', 'red', 'orange', 'purple'])
plt.title('Per-Class F1 Scores', fontsize=12, fontweight='bold')
plt.xlabel('Classes')
plt.ylabel('F1 Score')
plt.xticks(range(len(class_names)), class_names, rotation=45)
plt.grid(True, alpha=0.3)

for bar, score in zip(bars, f1_scores):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{score:.3f}', ha='center', va='bottom', fontsize=10)

# Feature importance visualization (morphology features)
plt.subplot(3, 4, 5)
feature_names = [
    'Mean', 'Std', 'Var', 'Skew', 'Kurt', 'Max', 'Min', 'P2P',
    'N_Peaks', 'R_Amp', 'R_Pos', 'QRS_Dur', 'QRS_W', 'QRS_Area',
    'P_Max', 'P_Min', 'P_Area', 'T_Max', 'T_Min', 'T_Area',
    'Slope+', 'Slope-', 'Slope_Avg', 'Slope_Var',
    'Pre_R', 'Post_R', 'Energy', 'Norm_E', 'RMS',
    'ZC', 'QRS_P', 'QRS_T', 'Beat_Var', 'Symmetry', 'Time2Peak', 'Up_Area', 'Down_Area'
]
# Simple feature importance based on standard deviation
feature_importance = np.std(x_train_morph_scaled, axis=0)
top_features_idx = np.argsort(feature_importance)[-10:]  # Top 10 features
plt.barh(range(len(top_features_idx)), feature_importance[top_features_idx])
plt.yticks(range(len(top_features_idx)), [feature_names[i] for i in top_features_idx])
plt.title('Top 10 Morphology Features', fontsize=12, fontweight='bold')
plt.xlabel('Importance (Std Dev)')

# Model architecture summary
plt.subplot(3, 4, 6)
plt.text(0.1, 0.9, 'Hybrid Model Summary:', fontsize=14, fontweight='bold', transform=plt.gca().transAxes)
plt.text(0.1, 0.8, f'Total Parameters: {model.count_params():,}', fontsize=12, transform=plt.gca().transAxes)
plt.text(0.1, 0.7, f'Morphology Features: {x_train_morph_scaled.shape[1]}', fontsize=12, transform=plt.gca().transAxes)
plt.text(0.1, 0.6, f'Raw Signal Length: {x_train_raw.shape[1]}', fontsize=12, transform=plt.gca().transAxes)
plt.text(0.1, 0.5, f'Test Accuracy: {test_score[1]:.4f}', fontsize=12, transform=plt.gca().transAxes)
plt.text(0.1, 0.4, f'Macro F1-Score: {f1_score(y_true_label, y_pred_label, average="macro"):.4f}', fontsize=12, transform=plt.gca().transAxes)
plt.text(0.1, 0.3, f'Training Epochs: {len(history_result.history["loss"])}', fontsize=12, transform=plt.gca().transAxes)
plt.text(0.1, 0.2, 'No Wavelet Transform', fontsize=12, transform=plt.gca().transAxes)
plt.axis('off')

# Precision and Recall plots
if 'precision' in history_result.history:
    plt.subplot(3, 4, 7)
    plt.plot(history_result.history['precision'], label='Training Precision', linewidth=2)
    plt.plot(history_result.history['val_precision'], label='Validation Precision', linewidth=2)
    plt.title('Model Precision', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid(True, alpha=0.3)

if 'recall' in history_result.history:
    plt.subplot(3, 4, 8)
    plt.plot(history_result.history['recall'], label='Training Recall', linewidth=2)
    plt.plot(history_result.history['val_recall'], label='Validation Recall', linewidth=2)
    plt.title('Model Recall', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.legend()
    plt.grid(True, alpha=0.3)

# Class distribution
plt.subplot(3, 4, 9)
unique, counts = np.unique(y_true_label, return_counts=True)
plt.pie(counts, labels=[class_names[i] for i in unique], autopct='%1.1f%%', startangle=90)
plt.title('Test Set Class Distribution', fontsize=12, fontweight='bold')

# Model comparison metrics
plt.subplot(3, 4, 10)
metrics = ['Train Acc', 'Val Acc', 'Test Acc', 'Macro F1', 'Weighted F1']
values = [train_score[1], val_score[1], test_score[1], 
          f1_score(y_true_label, y_pred_label, average='macro'),
          f1_score(y_true_label, y_pred_label, average='weighted')]
bars = plt.bar(metrics, values, color=['lightblue', 'lightgreen', 'lightcoral', 'gold', 'lightpink'])
plt.title('Model Performance Metrics', fontsize=12, fontweight='bold')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

for bar, value in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{value:.3f}', ha='center', va='bottom', fontsize=10)

# Morphology features description
plt.subplot(3, 4, 11)
plt.text(0.5, 0.5, 'Morphology Features (37):\n\n'
                   '• Statistical (8): Mean, Std, etc.\n'
                   '• Peak Detection (4): R-peak info\n'
                   '• QRS Complex (2): Duration, Area\n'
                   '• P & T Waves (6): Amplitudes, Areas\n'
                   '• Slopes (4): Gradient features\n'
                   '• Intervals (2): Time segments\n'
                   '• Energy (3): Signal power\n'
                   '• Clinical (8): Ratios, Symmetry\n\n'
                   'No Wavelet Preprocessing',
         ha='center', va='center', fontsize=10, transform=plt.gca().transAxes,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7))
plt.title('Feature Engineering', fontsize=12, fontweight='bold')
plt.axis('off')

# Training vs Validation gap
plt.subplot(3, 4, 12)
acc_gap = np.array(history_result.history['accuracy']) - np.array(history_result.history['val_accuracy'])
plt.plot(acc_gap, linewidth=2, color='red')
plt.title('Training-Validation Accuracy Gap', fontsize=12, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Accuracy Gap')
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# Save final model
model.save('./model/morphology_cnn_final.h5')
print("Morphology + CNN model saved successfully!")

# Save detailed results
import json
results_summary = {
    'model_type': 'Morphology + CNN Hybrid (No Wavelet)',
    'total_parameters': int(model.count_params()),
    'morphology_features': int(x_train_morph_scaled.shape[1]),
    'raw_signal_length': int(x_train_raw.shape[1]),
    'training_epochs': len(history_result.history['loss']),
    'test_accuracy': float(test_score[1]),
    'test_precision': float(test_score[2]),
    'test_recall': float(test_score[3]),
    'macro_f1': float(f1_score(y_true_label, y_pred_label, average='macro')),
    'weighted_f1': float(f1_score(y_true_label, y_pred_label, average='weighted')),
    'confusion_matrix': cm.tolist(),
    'classification_report': report,
    'morphology_feature_categories': {
        'statistical_features': 8,
        'peak_features': 4,
        'qrs_features': 2,
        'wave_features': 6,
        'slope_features': 4,
        'interval_features': 2,
        'energy_features': 3,
        'clinical_features': 8,
        'total_morphology_features': 37
    },
    'preprocessing': {
        'wavelet_transform': False,
        'feature_normalization': True,
        'raw_signal_preprocessing': 'median_filter_baseline_removal'
    }
}

with open('./model/morphology_cnn_results.json', 'w') as f:
    json.dump(results_summary, f, indent=2)

print(f"Training completed!")
print(f"Best validation accuracy: {max(history_result.history['val_accuracy']):.4f}")
print(f"Final test accuracy: {test_score[1]:.4f}")
print(f"Results saved to ./model/morphology_cnn_results.json")