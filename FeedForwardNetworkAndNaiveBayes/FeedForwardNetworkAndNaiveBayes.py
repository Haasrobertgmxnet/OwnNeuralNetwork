import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.naive_bayes import GaussianNB
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Laden des Iris-Datasets
data = load_iris()
X = data.data
y = data.target
class_names = data.target_names

# Train-Test-Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 1. Naive-Bayes-Modell
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)

# Naive-Bayes Ergebnisse
print("Naive Bayes Ergebnisse")
print(f"Accuracy: {accuracy_score(y_test, y_pred_nb):.2f}")
print(classification_report(y_test, y_pred_nb, target_names=class_names))

# 2. Multilayer-Perceptron-Modell mit TensorFlow/Keras
# One-Hot-Encoding der Labels für MLP
encoder = OneHotEncoder(sparse_output=False)
y_train_oh = encoder.fit_transform(y_train.reshape(-1, 1))
y_test_oh = encoder.transform(y_test.reshape(-1, 1))

# Standardisierung der Features für MLP
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Aufbau des MLP-Modells
mlp_model = Sequential([
    Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(8, activation='relu'),
    Dense(len(class_names), activation='softmax')
])
mlp_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Training
mlp_model.fit(X_train_scaled, y_train_oh, epochs=50, batch_size=8, verbose=0)

# Test
mlp_loss, mlp_accuracy = mlp_model.evaluate(X_test_scaled, y_test_oh, verbose=0)
y_pred_mlp = np.argmax(mlp_model.predict(X_test_scaled), axis=1)

# Multilayer-Perceptron Ergebnisse
print("\nMultilayer Perceptron Ergebnisse")
print(f"Accuracy: {mlp_accuracy:.2f}")
print(classification_report(y_test, y_pred_mlp, target_names=class_names))

# Konfusionsmatrixen plotten
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
cm_nb = confusion_matrix(y_test, y_pred_nb)
cm_mlp = confusion_matrix(y_test, y_pred_mlp)

ConfusionMatrixDisplay(cm_nb, display_labels=class_names).plot(ax=axes[0], cmap='Blues')
axes[0].set_title('Naive Bayes Confusion Matrix')

ConfusionMatrixDisplay(cm_mlp, display_labels=class_names).plot(ax=axes[1], cmap='Blues')
axes[1].set_title('MLP Confusion Matrix')

plt.tight_layout()
plt.show()

