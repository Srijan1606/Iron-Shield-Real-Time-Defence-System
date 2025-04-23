# Iron-Shield-Real-Time-Defence-System
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix, f1_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import keras_tuner as kt
import time

# Step 1: Generate enhanced synthetic data
def generate_trajectory_data(num_samples, R=10, max_dist=50, max_vel=5, noise_std_range=(0.1, 0.5)):
    data = []
    labels = []
    g = 9.8 / 1000
    target_intercept = num_samples // 2
    target_ignore = num_samples // 2
    intercept_count = 0
    ignore_count = 0

    while intercept_count < target_intercept or ignore_count < target_ignore:
        x0 = np.random.uniform(-max_dist, max_dist)
        y0 = np.random.uniform(-max_dist, max_dist)
        z0 = np.random.uniform(R, 2*max_dist)
        vx = np.random.uniform(-max_vel, max_vel)
        vy = np.random.uniform(-max_vel, max_vel)
        vz = np.random.uniform(-2, -0.5)
        noise_std = np.random.uniform(*noise_std_range)

        t = np.linspace(0, 10, 100)
        x = x0 + vx * t + np.random.normal(0, noise_std, len(t))
        y = y0 + vy * t + np.random.normal(0, noise_std, len(t))
        z = z0 + vz * t - 0.5 * g * t**2 + np.random.normal(0, noise_std, len(t))

        intersects = False
        for i in range(len(t)):
            dist = np.sqrt(x[i]**2 + y[i]**2 + z[i]**2)
            if dist <= R and z[i] >= 0:
                intersects = True
                break

        if intersects and intercept_count < target_intercept:
            data.append([x0, y0, z0, vx, vy, vz])
            labels.append(1)
            intercept_count += 1
        elif not intersects and ignore_count < target_ignore:
            data.append([x0, y0, z0, vx, vy, vz])
            labels.append(0)
            ignore_count += 1

    print(f"Class distribution: Intercept={intercept_count}, Ignore={ignore_count}")
    return np.array(data), np.array(labels)

# Generate dataset
num_samples = 20000
R = 10
X, y = generate_trajectory_data(num_samples, R, noise_std_range=(0.1, 0.5))

# Step 2: Normalize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Step 3: Split the data
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)

# Step 4: Build model with keras-tuner
def build_model(hp):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(units=hp.Int('units1', min_value=32, max_value=128, step=32), activation='relu', input_shape=(6,)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(units=hp.Int('units2', min_value=16, max_value=64, step=16), activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(units=hp.Int('units3', min_value=8, max_value32, step=8), activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    
    model.compile(optimizer=tf.keras.optimizers.Adam(hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

tuner = kt.RandomSearch(build_model,
                        objective='val_accuracy',
                        max_trials=10,
                        directory='tuner_dir',
                        project_name='missile_interception')
tuner.search(X_train, y_train, epochs=50, validation_data=(X_val, y_val),
             callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)])
best_model = tuner.get_best_models(num_models=1)[0]

# Step 5: Evaluate the model
test_loss, test_accuracy = best_model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_accuracy:.4f}")

# Step 6: Optimize threshold using F1 score
y_pred_proba = best_model.predict(X_test)

# Calibrate probabilities
calibrator = CalibratedClassifierCV(LogisticRegression(), method='sigmoid', cv=5)
y_pred_proba_calibrated = calibrator.fit(y_pred_proba, y_test).predict_proba(y_pred_proba)[:, 1]

precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba_calibrated)
f1_scores = [f1_score(y_test, y_pred_proba_calibrated >= t) for t in thresholds]
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]
print(f"Optimal Threshold (F1 Score): {optimal_threshold:.4f}")

# Step 7: Confusion matrix
y_pred = (y_pred_proba_calibrated >= optimal_threshold).astype(int)
cm = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix (Threshold {optimal_threshold:.4f}):\n{cm}")

# Step 8: Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

interpreter = tf.lite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Step 9: Test with new missile and post-processing
def check_trajectory(x0, y0, z0, vx, vy, vz, R=10, t_max=10):
    t = np.linspace(0, t_max, 100)
    x = x0 + vx * t
    y = y0 + vy * t
    z = z0 + vz * t - 0.5 * (9.8 / 1000) * t**2
    dist = np.sqrt(x**2 + y**2 + z**2)
    return np.min(dist) > R

new_missile = np.array([[15, 0, 20, -2, 0, -1]], dtype=np.float32)
new_missile_scaled = scaler.transform(new_missile).astype(np.float32)

start_time = time.time()
interpreter.set_tensor(input_details[0]['index'], new_missile_scaled)
interpreter.invoke()
prediction = interpreter.get_tensor(output_details[0]['index'])
inference_time = (time.time() - start_time) * 1000

if check_trajectory(*new_missile[0]):
    decision = "Ignore"  # Override if trajectory doesn’t intersect
else:
    decision = "Intercept" if prediction[0] >= optimal_threshold else "Ignore"

print(f"\nNew Missile Prediction: {prediction[0]:.4f} -> Decision: {decision} (Threshold: {optimal_threshold:.4f})")
print(f"Inference Time: {inference_time:.2f} ms")
