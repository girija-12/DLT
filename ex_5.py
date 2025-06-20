# Step 1: Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models
import tensorflow as tf

# Step 2: Load LFW dataset (filtering for people with at least 70 images)
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
X = lfw_people.images
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = len(target_names)

# Step 3: Preprocess and split dataset
X = X[..., np.newaxis] / 255.0  # Normalize and add channel dimension
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

y_train_cat = to_categorical(y_train, num_classes=n_classes)
y_test_cat  = to_categorical(y_test, num_classes=n_classes)

# Step 4: Build CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=X.shape[1:]),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(n_classes, activation='softmax')
])

# Step 5: Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Step 6: Train the model
history = model.fit(X_train, y_train_cat, epochs=10, batch_size=32,
                    validation_data=(X_test, y_test_cat))

# Step 7: Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test_cat)
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

# Step 8: Predict and visualize sample results
def visualize_predictions(model, X_test, y_test, target_names, num_images=5):
    preds = model.predict(X_test)
    pred_labels = np.argmax(preds, axis=1)

    plt.figure(figsize=(15, 4))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(X_test[i].squeeze(), cmap='gray')
        true_name = target_names[y_test[i]]
        pred_name = target_names[pred_labels[i]]
        color = 'green' if y_test[i] == pred_labels[i] else 'red'
        plt.title(f"True: {true_name}\nPred: {pred_name}", color=color)
        plt.axis('off')
    plt.show()

visualize_predictions(model, X_test, y_test, target_names)