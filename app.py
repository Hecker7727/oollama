import streamlit as st
import mysql.connector
from tensorflow.keras.applications import MobileNetV2, ResNet50, VGG16, InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess, decode_predictions as mobilenet_decode
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess, decode_predictions as resnet_decode
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess, decode_predictions as vgg16_decode
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess, decode_predictions as inception_decode
import numpy as np
import bcrypt
import os
import logging
import matplotlib.pyplot as plt
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import requests

# Set up logging
logging.basicConfig(level=logging.INFO)

# Connect to MySQL database using environment variables
db_connection = mysql.connector.connect(
    host=os.getenv("DB_HOST", "localhost"),
    user=os.getenv("DB_USER", "root"),
    passwd=os.getenv("DB_PASSWORD", "54321"),
    database=os.getenv("DB_NAME", "image_recognition")
)
cursor = db_connection.cursor()

# Function to hash a password
def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

# Function to register a new user
def register_user(username, password):
    try:
        hashed_password = hash_password(password)
        cursor.execute("INSERT INTO users (username, password_hash) VALUES (%s, %s)", (username, hashed_password.decode('utf-8')))
        db_connection.commit()
        return {"status": "success", "message": "Registration successful!"}
    except mysql.connector.Error as err:
        logging.error(f"Registration failed: {err}")
        return {"status": "error", "message": f"Registration failed: {err}"}

# Function to authenticate a user
def authenticate_user(username, password):
    try:
        cursor.execute("SELECT id, password_hash FROM users WHERE username = %s", (username,))
        result = cursor.fetchone()

        if result:
            user_id, stored_hash = result
            if bcrypt.checkpw(password.encode('utf-8'), stored_hash.encode('utf-8')):
                return {"status": "success", "user_id": user_id, "username": username}
            else:
                return {"status": "error", "message": "Invalid username or password."}
        else:
            return {"status": "error", "message": "Invalid username or password."}
    except mysql.connector.Error as err:
        logging.error(f"Authentication failed: {err}")
        return {"status": "error", "message": f"Authentication failed: {err}"}

# Function to update user profile
def update_user_profile(user_id, new_username, new_password):
    try:
        hashed_password = hash_password(new_password)
        cursor.execute("UPDATE users SET username = %s, password_hash = %s WHERE id = %s", (new_username, hashed_password.decode('utf-8'), user_id))
        db_connection.commit()
        return {"status": "success", "message": "Profile updated successfully!"}
    except mysql.connector.Error as err:
        logging.error(f"Profile update failed: {err}")
        return {"status": "error", "message": f"Profile update failed: {err}"}

# Function to perform image recognition and save results to MySQL
@st.cache_data
def recognize_image(image_file, user_id, image_name, model_name='MobileNetV2'):
    try:
        if model_name == 'MobileNetV2':
            model = MobileNetV2(weights='imagenet')
            preprocess_function = mobilenet_preprocess
            decode_function = mobilenet_decode
        elif model_name == 'ResNet50':
            model = ResNet50(weights='imagenet')
            preprocess_function = resnet_preprocess
            decode_function = resnet_decode
        elif model_name == 'VGG16':
            model = VGG16(weights='imagenet')
            preprocess_function = vgg16_preprocess
            decode_function = vgg16_decode
        elif model_name == 'InceptionV3':
            model = InceptionV3(weights='imagenet')
            preprocess_function = inception_preprocess
            decode_function = inception_decode
        else:
            raise ValueError("Invalid model selected.")

        # Read the image file from the BytesIO object
        img = Image.open(image_file)
        
        # Convert image to RGB if it has an alpha channel
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize image to the required size for the selected model
        if model_name == 'InceptionV3':
            img = img.resize((299, 299))
        else:
            img = img.resize((224, 224))
        
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_function(img_array)

        predictions = model.predict(img_array)
        decoded_predictions = decode_function(predictions, top=3)[0]

        # Extract predictions
        pred_labels = [pred[1] for pred in decoded_predictions]
        pred_confidences = [float(pred[2]) for pred in decoded_predictions]  # Convert to Python float

        # Prepare results as string
        result_text = "\n".join([f"{pred[1]}: {pred[2]*100:.2f}%" for pred in decoded_predictions])

        # Save predictions to database
        cursor.execute('''
            INSERT INTO image_predictions (user_id, image_name, model_used, prediction_1, prediction_2, prediction_3, confidence_1, confidence_2, confidence_3)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ''', (user_id, image_name, model_name, pred_labels[0], pred_labels[1], pred_labels[2], pred_confidences[0], pred_confidences[1], pred_confidences[2]))
        db_connection.commit()

        return {"status": "success", "predictions": decoded_predictions, "result_text": result_text}
    except Exception as e:
        logging.error(f"Image recognition failed: {e}")
        return {"status": "error", "message": str(e)}

# Function to generate image description
@st.cache_data
def generate_image_description(image_file, user_id, image_name):
    try:
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

        # Move model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Read the image file from the BytesIO object
        img = Image.open(image_file)
        inputs = processor(images=img, return_tensors="pt").to(device)  # Move input tensors to GPU
        pixel_values = inputs.pixel_values
        output_ids = model.generate(pixel_values, max_length=16, num_beams=4, no_repeat_ngram_size=2, early_stopping=True)
        description = processor.decode(output_ids[0], skip_special_tokens=True)

        # Save description to database
        cursor.execute('''
            UPDATE image_predictions
            SET image_description = %s
            WHERE user_id = %s AND image_name = %s
            ''', (description, user_id, image_name))
        db_connection.commit()

        return {"status": "success", "description": description}
    except Exception as e:
        logging.error(f"Image description generation failed: {e}")
        return {"status": "error", "message": str(e)}

# Function to display a pie chart
def display_pie_chart(predictions):
    labels = [pred[1] for pred in predictions]
    sizes = [pred[2] for pred in predictions]

    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig)

# Function to display a bar chart
def display_bar_chart(predictions):
    labels = [pred[1] for pred in predictions]
    sizes = [pred[2] for pred in predictions]

    fig, ax = plt.subplots()
    ax.bar(labels, sizes, color='skyblue')
    ax.set_xlabel('Predictions')
    ax.set_ylabel('Confidence')
    ax.set_title('Prediction Confidence Levels')
    st.pyplot(fig)

# Function to view user's past recognition results
def view_user_results(user_id):
    cursor.execute("SELECT image_name, prediction_1, confidence_1, prediction_2, confidence_2, prediction_3, confidence_3, image_description FROM image_predictions WHERE user_id = %s", (user_id,))
    results = cursor.fetchall()
    if results:
        st.subheader("Past Recognition Results")
        for result in results:
            st.write(f"Image Name: {result[0]}")
            st.write(f"1. {result[1]}: {result[2]*100:.2f}%")
            st.write(f"2. {result[3]}: {result[4]*100:.2f}%")
            st.write(f"3. {result[5]}: {result[6]*100:.2f}%")
            st.write(f"Description: {result[7]}")
            st.write("---")
    else:
        st.write("No past results found.")

# Function to apply the selected theme
def apply_theme(theme):
    if theme == "Dark":
        st.markdown(
            """
            <style>
            body {
                background-color: #2e2e2e;
                color: white;
            }
            .stApp {
                background-color: #2e2e2e;
                color: white;
            }
            .stTextInput>div>div>input, .stTextInput>div>div>textarea, .stTextInput>div>div>div>div>input {
                background-color: #333;
                color: white;
            }
            .stButton>button {
                background-color: #555;
                color: white;
            }
            .stDataFrame {
                background-color: #333;
                color: white;
            }
            .stSidebar {
                background-color: #333;
                color: white;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <style>
            body {
                background-color: white;
                color: black;
            }
            .stApp {
                background-color: white;
                color: black;
            }
            .stTextInput>div>div>input, .stTextInput>div>div>textarea, .stTextInput>div>div>div>div>input {
                background-color: white;
                color: black;
            }
            .stButton>button {
                background-color: #ddd;
                color: black;
            }
            .stDataFrame {
                background-color: white;
                color: black;
            }
            .stSidebar {
                background-color: #f5f5f5;
                color: black;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

# Main Streamlit app
def main():
    st.set_page_config(page_title="", layout="wide")

    # Define session state variables for user authentication
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "user_id" not in st.session_state:
        st.session_state.user_id = None
    if "username" not in st.session_state:
        st.session_state.username = None

    # Sidebar with theme selection
    st.sidebar.title("Settings")
    theme = st.sidebar.radio("Select Theme", ("Light", "Dark"))
    apply_theme(theme)

    # Main interface with options for Home, Login, Register, and Profile
    st.title("Image Recognition and Description App")

    menu = ["Home", "Login", "Register", "Profile"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Home")
        st.write("Welcome to the Image Recognition and Description App!")

    elif choice == "Login":
        st.subheader("Login")

        if st.session_state.authenticated:
            st.success(f"Logged in as {st.session_state.username}")

            # Image upload and recognition
            st.subheader("Image Recognition")
            model_name = st.selectbox("Select Model", ["MobileNetV2", "ResNet50", "VGG16", "InceptionV3"])
            image_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

            if image_file is not None:
                image_name = image_file.name
                st.image(image_file, caption=image_name)

                if st.button("Recognize Image"):
                    result = recognize_image(image_file, st.session_state.user_id, image_name, model_name)
                    if result["status"] == "success":
                        st.success("Image recognized successfully!")
                        st.write(result["result_text"])
                        display_pie_chart(result["predictions"])
                        display_bar_chart(result["predictions"])
                    else:
                        st.error(result["message"])

                if st.button("Generate Description"):
                    result = generate_image_description(image_file, st.session_state.user_id, image_name)
                    if result["status"] == "success":
                        st.success("Image description generated successfully!")
                        st.write(result["description"])
                    else:
                        st.error(result["message"])

            view_user_results(st.session_state.user_id)

        else:
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.button("Login"):
                result = authenticate_user(username, password)
                if result["status"] == "success":
                    st.session_state.authenticated = True
                    st.session_state.user_id = result["user_id"]
                    st.session_state.username = result["username"]
                    st.success("Login successful!")
                else:
                    st.error(result["message"])

    elif choice == "Register":
        st.subheader("Register")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Register"):
            result = register_user(username, password)
            if result["status"] == "success":
                st.success("Registration successful! Please log in.")
            else:
                st.error(result["message"])

    elif choice == "Profile":
        st.subheader("Profile")
        if st.session_state.authenticated:
            new_username = st.text_input("New Username", value=st.session_state.username)
            new_password = st.text_input("New Password", type="password")
            if st.button("Update Profile"):
                result = update_user_profile(st.session_state.user_id, new_username, new_password)
                if result["status"] == "success":
                    st.session_state.username = new_username
                    st.success("Profile updated successfully!")
                else:
                    st.error(result["message"])
        else:
            st.write("Please log in to update your profile.")

if __name__ == "__main__":
    main()