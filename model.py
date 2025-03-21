import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Define the Neural Network Model
class EmotionClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(EmotionClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Load Dataset Features
def load_data():
    data_dict = np.load("preprocessed_data.npy", allow_pickle=True).item()

    # Flatten MFCC features (40, 200) → (8000,)
    features = np.array([f.flatten() for f in data_dict["features"]], dtype=np.float32)  
    labels = np.array(data_dict["label"])

    # Debugging output
    print("Feature Shape:", features.shape)  # Should be (num_samples, 8000)
    print("Label Shape:", labels.shape)
    print("Labels Data:", labels)

    # Encode labels (e.g., "happy" → 0, "sad" → 1, etc.)
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)

    return train_test_split(features, labels, test_size=0.2, random_state=42), label_encoder.classes_

# Train the Model
def train():
    (X_train, X_test, y_train, y_test), class_names = load_data()

    input_size = 8000  # Set this to 40 (MFCC output size)
    hidden_size = 64
    num_classes = len(class_names)  # Based on emotions

    model = EmotionClassifier(input_size, hidden_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 50
    for epoch in range(epochs):
        inputs = torch.tensor(X_train, dtype=torch.float32)
        targets = torch.tensor(y_train, dtype=torch.long)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    # Save the model
    torch.save(model.state_dict(), "emotion_model.pth")
    np.save("label_classes.npy", class_names)  # ✅ Save label classes
    print("✅ Model trained and saved successfully!")

if __name__ == "__main__":
    train()
