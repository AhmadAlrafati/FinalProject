import os
import torch
import torch.nn as nn
import timm
from torchvision import transforms
from PIL import Image

# Define transformations (should match the transformations used during training)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define the model class (ensure it's the same as the one used in training)
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

# Function to load the trained model weights
def load_model(model_path):
    model = MyModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Function to predict if the mushroom is poisonous or edible
def predict_image(image_path, model):
    # Verify that the provided path is a valid image file
    if not os.path.isfile(image_path):
        raise ValueError(f"No file found at {image_path}. Please provide a valid image file path.")
    
    # Open and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add a batch dimension

    # Make a prediction
    with torch.no_grad():
        output = model(image)
        prediction = (output.squeeze() > 0.5).item()
        return "poisonous" if prediction else "edible"

# Main entry point for the program
if __name__ == "__main__":
    user_image_path = input("Enter the path to your mushroom image (JPG format): ")

    # Verify the input path ends with '.jpg'
    if os.path.exists(user_image_path) and user_image_path.lower().endswith('.jpg'):
        try:
            model_path = "C:/CSCI 127/135/245/my_trained_model.pth"
            model = load_model(model_path)
            prediction = predict_image(user_image_path, model)
            print(f"The mushroom is predicted to be {prediction}.")
        except Exception as e:
            print(f"An error occurred: {e}")
    else:
        print("Invalid image path or file format. Please enter a valid path to a '.jpg' file.")
