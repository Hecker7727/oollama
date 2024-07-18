import streamlit as st
import mysql.connector
from tensorflow.keras.applications import MobileNetV2, ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess, decode_predictions as mobilenet_decode
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess, decode_predictions as resnet_decode
import numpy as np
import bcrypt
import tempfile
import os
import logging
import matplotlib.pyplot as plt
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

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
        else:
            raise ValueError("Invalid model selected.")

        # Create a temporary file in the current working directory
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(image_file.read())
            temp_file.flush()  # Ensure all data is written
            temp_file_path = temp_file.name

        img = image.load_img(temp_file_path, target_size=(224, 224))
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

        os.remove(temp_file_path)  # Clean up temporary file
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

        image = Image.open(image_file)
        inputs = processor(images=image, return_tensors="pt")
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
                background-color: #444;
                color: white;
                border: 1px solid #555;
            }
            .stButton>button:hover {
                background-color: #555;
                color: white;
            }
            .stSelectbox>div>div>div>div {
                background-color: #333;
                color: white;
            }
            .stSelectbox>div>div>div>div:hover {
                background-color: #444;
                color: white;
            }
            .stCheckbox>div>div>div {
                background-color: #333;
                color: white;
            }
            .stCheckbox>div>div>div:hover {
                background-color: #444;
                color: white;
            }
            .stRadio>div>div>div {
                background-color: #333;
                color: white;
            }
            .stRadio>div>div>div:hover {
                background-color: #444;
                color: white;
            }
            .stTextInput>div>div>input:focus, .stTextInput>div>div>textarea:focus, .stTextInput>div>div>div>div>input:focus {
                border: 1px solid #555;
                outline: none;
            }
            </style>
            """, unsafe_allow_html=True
        )
    elif theme == "Light":
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
                background-color: #f0f0f0;
                color: black;
                border: 1px solid #ccc;
            }
            .stButton>button:hover {
                background-color: #e0e0e0;
                color: black;
            }
            .stSelectbox>div>div>div>div {
                background-color: white;
                color: black;
            }
            .stSelectbox>div>div>div>div:hover {
                background-color: #f0f0f0;
                color: black;
            }
            .stCheckbox>div>div>div {
                background-color: white;
                color: black;
            }
            .stCheckbox>div>div>div:hover {
                background-color: #f0f0f0;
                color: black;
            }
            .stRadio>div>div>div {
                background-color: white;
                color: black;
            }
            .stRadio>div>div>div:hover {
                background-color: #f0f0f0;
                color: black;
            }
            .stTextInput>div>div>input:focus, .stTextInput>div>div>textarea:focus, .stTextInput>div>div>div>div>input:focus {
                border: 1px solid #ccc;
                outline: none;
            }
            </style>
            """, unsafe_allow_html=True
        )

# Streamlit app layout
def main():
    st.set_page_config(page_title="Image Recognition App", layout="wide")
    
    # Theme selection
    theme = st.sidebar.selectbox("Select Theme", ["Light", "Dark"])
    apply_theme(theme)
    
    st.title("Image Recognition and Description App")
    st.sidebar.title("User Authentication")
    
    # Registration
    st.sidebar.subheader("Register")
    reg_username = st.sidebar.text_input("New Username", key="reg_username")
    reg_password = st.sidebar.text_input("New Password", type="password", key="reg_password")
    reg_button = st.sidebar.button("Register", key="register_button")
    
    if reg_button:
        reg_result = register_user(reg_username, reg_password)
        st.sidebar.write(reg_result["message"])
    
    # Authentication
    st.sidebar.subheader("Login")
    auth_username = st.sidebar.text_input("Username", key="auth_username")
    auth_password = st.sidebar.text_input("Password", type="password", key="auth_password")
    auth_button = st.sidebar.button("Login", key="auth_button")
    
    if auth_button:
        auth_result = authenticate_user(auth_username, auth_password)
        if auth_result["status"] == "success":
            st.session_state.authenticated = True
            st.session_state.user_id = auth_result["user_id"]
            st.session_state.username = auth_result["username"]
            st.sidebar.write(f"Welcome, {auth_result['username']}!")
        else:
            st.sidebar.write(auth_result["message"])
    
    # Main application
    if st.session_state.get("authenticated", False):
        st.sidebar.subheader("User Profile")
        new_username = st.sidebar.text_input("New Username", key="new_username")
        new_password = st.sidebar.text_input("New Password", type="password", key="new_password")
        update_button = st.sidebar.button("Update Profile", key="update_profile_button")
        
        if update_button:
            update_result = update_user_profile(st.session_state.user_id, new_username, new_password)
            st.sidebar.write(update_result["message"])
        
        st.sidebar.subheader("Actions")
        action = st.sidebar.radio("Select Action", ["Recognize Image", "Generate Description", "View Results"], key="action")
        
        if action == "Recognize Image":
            st.subheader("Upload Image for Recognition")
            image_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="image_file_recognition")
            model_name = st.selectbox("Select Model", ["MobileNetV2", "ResNet50"], key="model_name_recognition")
            image_name = st.text_input("Enter Image Name", key="image_name_recognition")
            if st.button("Recognize Image", key="recognize_button"):
                if image_file and image_name:
                    recognition_result = recognize_image(image_file, st.session_state.user_id, image_name, model_name)
                    if recognition_result["status"] == "success":
                        st.write("Recognition Results:")
                        st.write(recognition_result["result_text"])
                        st.write("Visualizations:")
                        display_pie_chart(recognition_result["predictions"])
                        display_bar_chart(recognition_result["predictions"])
                    else:
                        st.write(f"Error: {recognition_result['message']}")
                else:
                    st.write("Please upload an image and enter an image name.")
        
        elif action == "Generate Description":
            st.subheader("Upload Image for Description Generation")
            image_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="image_file_description")
            image_name = st.text_input("Enter Image Name", key="image_name_description")
            if st.button("Generate Description", key="generate_description_button"):
                if image_file and image_name:
                    description_result = generate_image_description(image_file, st.session_state.user_id, image_name)
                    if description_result["status"] == "success":
                        st.write("Generated Description:")
                        st.write(description_result["description"])
                    else:
                        st.write(f"Error: {description_result['message']}")
                else:
                    st.write("Please upload an image and enter an image name.")
        
        elif action == "View Results":
            st.subheader("View Your Past Recognition Results")
            view_user_results(st.session_state.user_id)
    
    else:
        st.write("Please log in to use the application.")

if __name__ == "__main__":
    main()




