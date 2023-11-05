import os
from flask import Flask, request, jsonify, render_template, send_from_directory

from local_settings import SECRET_KEY
import io, base64
from PIL import Image


import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

from PIL import Image
import numpy as np

import numpy as np
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from PIL import Image
import requests

import webcolors
from webcolors import rgb_to_name, CSS3_HEX_TO_NAMES
import numpy as np
# from scipy.stats import mode


import os
import openai

openai.api_key = "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"


class Clothing:
    def __init__(self, path, color, type):
        self.path = path
        self.color = color
        # self.fabric = fabric
        self.type = type
        # self.size = size

def path_classi(image_path):
    # Step 1: Load and preprocess an image
    img = Image.open(image_path)
    img = img.resize((224, 224))  # Resize the image to match VGG16's input size
    x = np.array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Step 2: Load the pre-trained VGG16 model
    model = VGG16(weights='imagenet')

    # Step 3: Preprocess the image to match the format expected by VGG16

    # Step 4: Make predictions using the pre-trained model
    predictions = model.predict(x)

    # Step 5: Decode and interpret the predictions
    decoded_predictions = decode_predictions(predictions, top=5)[0]

    return decoded_predictions[0][1]

app = Flask(__name__)

# Directory to store captured images
CAPTURED_IMAGES_DIR = "captured_images"

# Create the directory if it doesn't exist
os.makedirs(CAPTURED_IMAGES_DIR, exist_ok=True)

@app.route("/")
def hello():
    return render_template("camera.html")

@app.route('/camera')
def camera():
    return render_template("camera.html")

image_descriptions = {}

######## find closet_color
def find_closest_color(rgb_color):

    min_distance = float('inf')
    closest_color = None
    
    for css3_hex, color_name in CSS3_HEX_TO_NAMES.items():
        # Convert CSS3 hex color to RGB
        known_rgb = webcolors.hex_to_rgb(css3_hex)
        
        # Calculate Euclidean distance between the known and target colors
        distance = np.linalg.norm(np.array(known_rgb) - np.array(rgb_color))
        
        if distance < min_distance:
            min_distance = distance
            closest_color = color_name

    return closest_color

def calculate_mode(numbers):
    # Create a dictionary to store the frequency of each number
    frequency = {}

    # Find the frequency of each number in the list
    for number in numbers:
        if number in frequency:
            frequency[number] += 1
        else:
            frequency[number] = 1

    # Find the number(s) with the highest frequency
    max_frequency = max(frequency.values())
    mode = [number for number, freq in frequency.items() if freq == max_frequency]

    return mode
def frame_to_color(path):
    img = Image.open(path)
    img = np.array(img)
    # Calculate the dimensions of the middle square
    middle_size = min(img.shape[0], img.shape[1]) // 2

    # Calculate the coordinates of the top-left and bottom-right corners of the middle square
    top_left_x = (img.shape[1] - middle_size) // 2
    top_left_y = (img.shape[0] - middle_size) // 2
    bottom_right_x = top_left_x + middle_size
    bottom_right_y = top_left_y + middle_size

    cropped_photo = img[top_left_y:bottom_right_y, top_left_x:bottom_right_x, :]
    reshaped_image = cropped_photo.reshape(-1, 3)

    mode_color = calculate_mode(reshaped_image, axis=0)

    mode_color = mode_color.mode[0]

    single_color = np.full((100, 100, 3), mode_color, dtype=np.uint8)
    return find_closest_color(single_color)




########################

'''
@app.route('/mycloset')
def mycloset():
    return render_template("mycloset.html", data=image_descriptions)
'''

@app.route('/mycloset')
def mycloset():
    # Fetch weather data for Toronto
    city = 'Toronto'
    url_path = (
        "https://api.openweathermap.org/data/2.5/weather?q="
        + city
        + "&units=metric"
        + "&appid="
        + SECRET_KEY
    )

    r = requests.get(url_path)
    toronto_weather = r.json()

    return render_template("mycloset.html", data=image_descriptions, toronto_weather = toronto_weather)

@app.route('/store_image', methods=['POST'])
def store_image():
    try:
        data = request.get_json()
        image_data = data.get("image")

        # Generate a unique filename (you can use a more robust naming scheme)
        filename = os.path.join(CAPTURED_IMAGES_DIR, f"captured_image_{len(os.listdir(CAPTURED_IMAGES_DIR))}.jpg")

        # Save the captured image
        with open(filename, "wb") as file:
            file.write(base64.b64decode(image_data.split(",")[1]))  # Extract base64 data

        return jsonify({'message': 'Image stored successfully'})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/classify_image', methods=['POST'])
def classify_image():
    try:
        data = request.get_json()
        image_data = data.get("image")

        # Generate a unique filename (you can use a more robust naming scheme)
        filename = os.path.join(CAPTURED_IMAGES_DIR, f"captured_image_{len(os.listdir(CAPTURED_IMAGES_DIR))}.jpg")

        # Save the captured image
        with open(filename, "wb") as file:
            file.write(base64.b64decode(image_data.split(",")[1]))  # Extract base64 data

        # Run classification on the captured image
        image_description = path_classi(filename)
        #image_description = 'sweater'
        # image_color = frame_to_color(filename)
        image_color = 'blue'
        
        # Store the description in the dictionary along with the image path
        image_descriptions[filename] = [image_description, image_color]
        print(image_descriptions)
        #print(jsonify({'description': image_descriptions}))
        return jsonify({'description': image_description})
    
    except Exception as e:
        return jsonify({'error': str(e)})


def generate_question(clothes, temp, wind, precipitation):
    questions = "I have "
    for cloth in clothes:
        questions = questions + cloth.color + " " + cloth.type + ", "
    

    questions = questions + ". Today's tempurature is " + str(temp) + " degrees celsius, the wind is " + str(wind) + ", the precipitations is " + str(precipitation) + "percent. Please choose what to wear according to what I have"
    print(questions)
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "system", "content": questions}],
    temperature=0,
    max_tokens=1024
    )

    response =  response["choices"][0]["message"]["content"]
    
    response = response.split('. ')
    response_lst = []
    for word in response:
        if ": " in word:
            response_lst.append(word[:word.index(": ")])

    if len(response_lst) == 0:
        return "unable to give you suggestions, wear whatever you have"
    return response_lst


@app.route('/qlouni')
def qlouni():
    clothes = []
    for k, v in image_descriptions.items():
        k = Clothing(k, v[0], v[1])
        clothes.append(k)
    # Retrieve temperature, wind, and precipitation data from URL parameters
    temperature = request.args.get('temperature')
    wind = request.args.get('wind')
    precipitation = request.args.get('precipitation')

    # Generate the question and receive the response
    response = generate_question(clothes, temperature, wind, precipitation)
    return render_template("qlouni.html", temperature=temperature, wind=wind, precipitation=precipitation, response=response)


##############
#############



# testimg = openai.Image.create(
#     prompt="How can I style black dress pants?",
#     n=2,
#     size="1024x1024"
# )
# print(testimg)


@app.route('/captured_images/<filename>')
def serve_image(filename):
    return send_from_directory('captured_images', filename)

    
if __name__ == '__main__':
    app.run()
    #print(frame_to_color('archive/teetest.jpeg'))
    #print(generate_question([Clothing("k","Blue", "ack"), Clothing("k","Black", "rack"), Clothing("k","black", 'dress pants'), Clothing("k","grey", "hoodie"), Clothing("k","white", "hat"), Clothing("k","yellow", "raincoats")], 50, 3, 50))
    #print(modify_str(generate_question([clothing("Blue", "T-Shirt"), clothing("Black", "Sneakers"), clothing("black", 'dress pants'), clothing("grey", "hoodie"), clothing("white", "hat"), clothing("yellow", "raincoats")], 28, 3, 50, "work")))
    #image_descriptions = {'captured_images/captured_image_5.jpg': ['wok', 'blue'], 'captured_images/captured_image_6.jpg': ['hand_blower', 'blue']}
    #print("me")

@app.route('/send_data', methods=['POST'])
def send_data():
    try:
        # Get the data sent from camera.html
        data = request.get_json()

        # Send the data to mycloset.html
        return render_template("mycloset.html", data=data)
    except Exception as e:
        return jsonify({'error': str(e)})