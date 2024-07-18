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
                background-color: #555;
                color: white;
            }
            .stDataFrame {
                background-color: #333;
                color: white;
            }
            .stSidebar {
                background-color: #2e2e2e;
            }
            .css-1lcbmhc, .css-1d391kg, .css-1l02zno, .css-1aumxhk, .css-1aumxhk, .css-1aumxhk, .css-1aumxhk {
                color: white;
            }
            </style>
            """, unsafe_allow_html=True)
    else:
        st.markdown(
            """
            <style>
            body {
                background-color: #fff;
                color: black;
            }
            .stApp {
                background-color: #fff;
                color: black;
            }
            .stTextInput>div>div>input, .stTextInput>div>div>textarea, .stTextInput>div>div>div>div>input {
                background-color: #fff;
                color: black;
            }
            .stButton>button {
                background-color: #ddd;
                color: black;
            }
            .stDataFrame {
                background-color: #fff;
                color: black;
            }
            .stSidebar {
                background-color: #f8f9fa;
            }
            .css-1lcbmhc, .css-1d391kg, .css-1l02zno, .css-1aumxhk, .css-1aumxhk, .css-1aumxhk, .css-1aumxhk {
                color: black;
            }
            </style>
            """, unsafe_allow_html=True)

# Main Streamlit app
def main():
    st.title("Image Recognition and Description App")

    menu = ["Home", "Login", "Register"]
    choice = st.sidebar.selectbox("Menu", menu)
    theme = st.sidebar.radio("Choose Theme", ("Light", "Dark"))
    apply_theme(theme)

    # Maintain session state for logged-in user
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "user_id" not in st.session_state:
        st.session_state.user_id = None
    if "username" not in st.session_state:
        st.session_state.username = None

    if choice == "Home":
        st.subheader("Home")
        st.write("Please login or register to use the image recognition features.")

    elif choice == "Login":
        st.subheader("Login")

        username = st.text_input("Username")
        password = st.text_input("Password", type='password')

        if st.button("Login"):
            result = authenticate_user(username, password)
            if result["status"] == "success":
                st.session_state.logged_in = True
                st.session_state.user_id = result["user_id"]
                st.session_state.username = result["username"]
                st.success(f"Welcome {username}!")

        if st.session_state.logged_in:
            st.subheader("Image Recognition and Description")
            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

            if uploaded_file is not None:
                image_name = uploaded_file.name

                # Model selection
                model_name = st.selectbox("Select Model", ["MobileNetV2", "ResNet50"])

                if st.button("Recognize Image"):
                    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
                    st.write("Recognizing...")
                    result = recognize_image(uploaded_file, st.session_state.user_id, image_name, model_name)

                    if result["status"] == "success":
                        st.success("Image recognized successfully!")
                        st.write(result["result_text"])

                        # Display pie chart and bar chart
                        display_pie_chart(result["predictions"])
                        display_bar_chart(result["predictions"])
                    else:
                        st.error(result["message"])

                if st.button("Generate Description"):
                    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
                    st.write("Generating description...")
                    description_result = generate_image_description(uploaded_file, st.session_state.user_id, image_name)
                    if description_result["status"] == "success":
                        st.success("Image description generated successfully!")
                        st.write(description_result["description"])
                    else:
                        st.error(description_result["message"])

            # View user's past results
            view_user_results(st.session_state.user_id)
        else:
            st.error("Please login to access this feature.")

    elif choice == "Register":
        st.subheader("Register")

        new_username = st.text_input("New Username")
        new_password = st.text_input("New Password", type='password')

        if st.button("Register"):
            result = register_user(new_username, new_password)
            if result["status"] == "success":
                st.success(result["message"])
                st.info("Please go to the Login menu to login.")
            else:
                st.error(result["message"])

if __name__ == '__main__':
    main()