<div align="center">
  <img src="logo.png" alt="Logo" style="width:80%;"/>
</div>

<h1 align="center">🐶  Animal Classification with 10 Categories 🐱</h1>
<p align="center">Developed by <a href="https://www.linkedin.com/in/juancarlosgarzon/">Juan Carlos Garzon</a></p>

---

<div align="center">
  <img src="background.jpg" alt="Image" style="width:80%;"/>
</div>

---

<h3 align="center">Demo VIDEO</h3>



---

## Dataset

The dataset used for this project contains images of 10 different animal categories:

- Butterflies: 2112 images
- Cats: 1668 images
- Chickens: 3098 images
- Cows: 1866 images
- Dogs: 4860 images
- Elephants: 1446 images
- Horses: 2623 images
- Sheep: 1820 images
- Spiders: 4821 images
- Squirrels: 1862 images

---

## Project Goal

The main goal of this project is to create an animal classification system using a dataset containing images of 10 different animal categories. The project involves training two architectures from scratch: a custom model and an EfficientNetB4 model. Additionally, two models are trained using transfer learning: SwinTransformer and ConvNeXt. The trained models can be found in the `models` folder.

Once the training and fine-tuning were completed, a user-friendly **Streamlit** web interface was built, allowing users to upload images for immediate processing and classification results.

This project combines the latest techniques in deep learning and web interface design to provide an intuitive, easy-to-use solution for animal classification.

---

### Key Components:
- **Dataset**: Animal Classification with 10 Categories
- **Models**: Custom Model, EfficientNetB4, SwinTransformer, ConvNeXt (Trained models are in the `models` folder)
- **Web Interface**: Streamlit for intuitive image and video upload and processing

---

### Animals Classified:
- Butterflies
- Cats
- Chickens
- Cows
- Dogs
- Elephants
- Horses
- Sheep
- Spiders
- Squirrels

---

## Getting Started

To get started with the **Animal Classification with 10 Categories** project, follow the steps below. The models were trained using **Visual Studio Code**, running on **Linux Ubuntu**, with an **NVIDIA GeForce RTX 4080 GPU**.

### Prerequisites
Before running the application, ensure you have the following installed:
- **Miniforge/Mamba** (for creating and managing environments)
- **Visual Studio Code** (or any IDE for running Python)
- **NVIDIA Drivers** and **CUDA** (for GPU acceleration)

### Steps to Set Up

1. **Create a Conda environment**:

   Start by creating a Mamba environment. You can name it whatever you like (e.g., `animal_classification_env`). Open a terminal and run:

   ```bash
   conda create --name animal_classification_env python=3.10
    ```
   Replace animal_classification_env with your preferred environment name.

2. **Activate the environment**:

   Once the environment is created, activate it with:
  
   ```bash
   conda activate animal_classification_env
   ```

3. **Clone the repository**:

   Clone the repository from GitHub:

   ```bash
   git clone https://github.com/Juank0621/animal_10.git
   ```

4. **Install dependencies**:
    
   Next, install the necessary libraries by using the requirements.txt file. This file includes all the required dependencies for running the project. Run the following command:
    
   ```bash
   pip install -r requirements.txt
   ```
    
5. **Launch the Streamlit interface**:
    
   Finally, to interact with the model via a web interface, run the Streamlit application:
    
   ```bash
   streamlit run animal_app.py
   ```
   
   This will open a browser window with the interface, where you can upload images for classification.

---

## Screenshots

Below are screenshots of the project's user interface, showcasing the **Streamlit** web application:

### Streamlit Interface
The Streamlit application provides a user-friendly interface for uploading images for object detection.

![Streamlit Interface](streamlit.png)



