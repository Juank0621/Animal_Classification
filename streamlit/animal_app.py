import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from models import SwinTransformerTransferLearning, ConvNeXtTransferLearning, CustomModel, EfficientNetB1  # Import the model classes
import os

# Define the categories
CATEGORIES = ['Butterfly', 'Cat', 'Chicken', 'Cow', 'Dog', 'Elephant', 'Horse', 'Sheep', 'Spider', 'Squirrel']

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
        "EfficientNetB1": EfficientNetB1,
        "SwinTransformerTransferLearning": SwinTransformerTransferLearning,
        "ConvNeXtTransferLearning": ConvNeXtTransferLearning
    }

    # Select the appropriate model class based on the input model name
    model_class = model_classes[model_name]

    # Initialize the model
    model = model_class()

    # Verify if the model path exists
    if not os.path.exists(model_path):
        st.error(f"The model file {model_path} don't exist.")
        return None

    # Load the model weights, disable gradients, and prepare the model for inference
    try:
        with torch.no_grad():  # Disable gradient calculations to save memory during inference
            model.load_state_dict(torch.load(model_path, map_location=device))  # Load pretrained weights
            model.eval()  # Set the model to evaluation mode
            model.to(device)  # Move the model to the appropriate device (GPU or CPU)
            st.success(f"Model {model_name} loaded successfully.")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

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

    # Display the logo image
    st.image("/home/juangarzon/AI_Projects/animal_10/datasets/logo.png", use_container_width=True)

    st.title("🐶 Animal Classifier App 🐱")

    # Sidebar for selecting the model
    st.sidebar.title("🐻 Select Model 🐷")
    model_option = st.sidebar.radio("Choose a trained model:", ("CustomModel", "EfficientNetB1", "SwinTransformerTransferLearning", "ConvNeXtTransferLearning"))

    # Map model options to file paths
    model_paths = {
    "CustomModel": "/home/juangarzon/AI_Projects/animal_10/models/CustomModel.pt",
    "EfficientNetB1": "/home/juangarzon/AI_Projects/animal_10/models/EfficientNetB1.pt",
    "SwinTransformerTransferLearning": "/home/juangarzon/AI_Projects/animal_10/models/SwinTransformer.pt",
    "ConvNeXtTransferLearning": "/home/juangarzon/AI_Projects/animal_10/models/ConvNeXt.pt"
    }

    # Display information about supported categories
    st.markdown(
        f"""
        <div style="background-color: #f0f0f0; padding: 10px; border-radius: 5px;">
            🎓 This model can classify the following animals:<br>{', '.join([category.capitalize() for category in CATEGORIES])}
        </div>
        """,
        unsafe_allow_html=True
    )

    # Add space between sections
    st.markdown("<br>", unsafe_allow_html=True)

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
        st.markdown(
            f"""
            <div style="background-color: #d4edda; padding: 10px; border-radius: 5px;">
                The image is classified as: {predicted_category}
            </div>
            """,
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()


