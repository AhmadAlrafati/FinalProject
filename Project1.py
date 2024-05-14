import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import train_test_split
from torchvision import transforms
from PIL import Image
import timm
from sklearn.metrics import classification_report
import numpy as np
from collections import Counter

# Set a random seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Step 1: Specify the path to the dataset
dataset_folder = "C:/CSCI 127/135/245/Mushroom"

# Step 2: Define transformations with enhanced data augmentation
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Step 3: Define custom Dataset class
class CustomDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.image_paths = []
        self.labels = []
        for class_name in os.listdir(folder_path):
            class_folder = os.path.join(folder_path, class_name)
            for image_name in os.listdir(class_folder):
                if image_name.lower().endswith('.jpg'):
                    self.image_paths.append(os.path.join(class_folder, image_name))
                    self.labels.append(1 if class_name == 'poisonous' else 0)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# Step 4: Load dataset and print class distribution
dataset = CustomDataset(dataset_folder, transform=transform)
print(f"Original class distribution: {Counter(dataset.labels)}")

# Step 5: Stratified train-test split
indices = list(range(len(dataset)))
train_indices, test_indices = train_test_split(
    indices,
    test_size=0.2,
    random_state=42,
    stratify=dataset.labels
)

# Create data loaders with Subset to maintain class balance
train_dataset = Subset(dataset, train_indices)
test_dataset = Subset(dataset, test_indices)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Step 6: Define a simplified model architecture
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.model = timm.create_model('resnet18', pretrained=True, in_chans=3)
        self.model.reset_classifier(0)  # Remove the existing classifier
        self.fc = nn.Linear(self.model.num_features, 1)

    def forward(self, x):
        x = self.model(x)
        x = self.fc(x)
        return torch.sigmoid(x)

# Step 7: Initialize the model
model = MyModel()

# Step 8: Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 9: Training loop
num_epochs = 10
patience = 3  # Early stopping patience
best_loss = float('inf')
patience_counter = 0

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()  # Zero the parameter gradients
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, labels.unsqueeze(1).float())  # Compute the loss
        loss.backward()  # Backward pass
        optimizer.step()  # Optimize
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(train_dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    # Early stopping condition
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

print('Finished Training')

# Step 10: Evaluation
model.eval()  # Set the model to evaluation mode
all_predictions = []
all_labels = []
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        predictions = (outputs.squeeze() > 0.5).int().tolist()  # Apply threshold for binary classification
        all_predictions.extend(predictions)
        all_labels.extend(labels.tolist())

# Step 11: Print classification report
unique_labels = np.unique(all_labels + all_predictions)
label_names = ['edible', 'poisonous']
print(classification_report(all_labels, all_predictions, labels=unique_labels, target_names=[label_names[i] for i in unique_labels]))

# Save the trained model
model_path = "C:/CSCI 127/135/245/my_trained_model.pth"
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

