import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from models import SwinTransformerTransferLearning, ConvNeXtTransferLearning, CustomModel, EfficientNetB4  # Import the model classes

# Define the categories
CATEGORIES = ['butterfly', 'cat', 'chicken', 'cow', 'dog', 'elephant', 'horse', 'sheep', 'spider', 'squirrel']

# Function to add a background image to the Streamlit app
def add_background(image_path):
    css = f"""
    <style>
    .stApp {{
        background: url(data:image/png;base64,{get_base64_of_bin_file(image_path)}) no-repeat center center fixed;
        background-size: contain; /* Adjust the image to fit the window */
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Helper function to convert an image to base64 format
def get_base64_of_bin_file(bin_file):
    import base64
    with open(bin_file, "rb") as file:
        encoded = base64.b64encode(file.read()).decode()
    return encoded

@st.cache_resource  # Cache the model to avoid reloading it repeatedly for performance optimization
def load_model(model_name, model_path):
    torch.cuda.empty_cache()  # Free up GPU memory to prevent out-of-memory errors

    # Delete the existing model from memory, if it exists, to avoid memory leaks
    if 'model' in globals():
        del model
        torch.cuda.empty_cache()

    # Determine whether to use the GPU (if available) or CPU for inference
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Map model names to their corresponding classes
    model_classes = {
        "CustomModel": CustomModel,
        "EfficientNetB4": EfficientNetB4,
        "SwinTransformerTransferLearning": SwinTransformerTransferLearning,
        "ConvNeXtTransferLearning": ConvNeXtTransferLearning
    }

    # Select the appropriate model class based on the input model name
    model_class = model_classes[model_name]

    # If the model requires 'num_classes' as an argument, pass the number of categories
    if model_name in ["ConvNeXtTransferLearning", "SwinTransformerTransferLearning"]:
        model = model_class(num_classes=len(CATEGORIES))  # Initialize the model with class count
    else:
        model = model_class()  # For models that don't require 'num_classes'

    # Load the model weights, disable gradients, and prepare the model for inference
    with torch.no_grad():  # Disable gradient calculations to save memory during inference
        model.load_state_dict(torch.load(model_path, map_location=device))  # Load pretrained weights
        model.eval()  # Set the model to evaluation mode
        model.to(device)  # Move the model to the appropriate device (GPU or CPU)

    return model  # Return the loaded and ready-to-use model

# Function to classify an image
def classify_image(model, image):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available

    # Define the transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize image to 224x224 (default for most models)
        transforms.ToTensor(),  # Convert PIL Image to Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize to ImageNet standards
    ])

    # Preprocess the image
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to GPU if available

    # Perform inference with mixed precision
    with torch.amp.autocast('cuda'):
        with torch.no_grad():
            outputs = model(image)
            _, predicted = outputs.max(1)  # Get the index of the highest score

    # Free up GPU memory
    torch.cuda.empty_cache()

    return CATEGORIES[predicted.item()]  # Return the corresponding category

# Main Streamlit app
def main():

    # Add custom CSS for background
    add_background("background.jpg")

    st.title("🐶 Animal Classifier App 🐱")

    # Sidebar for selecting the model
    st.sidebar.title("🐻 Select Model 🐷")
    model_option = st.sidebar.radio("Choose a trained model:", ("CustomModel", "EfficientNetB4", "SwinTransformerTransferLearning", "ConvNeXtTransferLearning"))

    # Map model options to file paths
    model_paths = {
    "CustomModel": "/home/juangarzon/AI_Projects/animal_10/models/CustomModel.pt",
    "EfficientNetB4": "/home/juangarzon/AI_Projects/animal_10/models/EfficientNetB4.pt",
    "SwinTransformerTransferLearning": "/home/juangarzon/AI_Projects/animal_10/models/SwinTransformer.pt",
    "ConvNeXtTransferLearning": "/home/juangarzon/AI_Projects/animal_10/models/ConvNeXt.pt"
    }

    # Display information about supported categories
    st.info(f"🎓 **This model can classify the following animals:**\n\n{', '.join([category.capitalize() for category in CATEGORIES])}")

    # Load the selected model
    selected_model_path = model_paths[model_option]
    model = load_model(model_option, selected_model_path)

    # Upload image section
    st.sidebar.title("📸 Upload an Image")
    uploaded_file = st.sidebar.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        image = Image.open(uploaded_file).convert("RGB")

        # Classify the image
        st.header("🎯 Prediction")
        predicted_category = classify_image(model, image)
        st.success(f"The image is classified as: **{predicted_category}**")

if __name__ == "__main__":
    main()


