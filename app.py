from flask import Flask, render_template, request, redirect, url_for
import numpy as np
from PIL import Image
import io
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.applications.xception import Xception
import pickle

app = Flask(__name__)

def extract_features(img, model):
    try:
        img = img.resize((299, 299))
        img_array = np.array(img)
        if img_array.shape[2] == 4:
            img_array = img_array[..., :3]
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 127.5
        img_array = img_array - 1.0
        feature = model.predict(img_array)
        return feature
    except Exception as e:
        print("ERROR:", e)
        return None

def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'start'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([photo, sequence], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break
    return in_text

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}

xception_model = Xception(include_top=False, pooling="avg")
# Load the tokenizer
with open("tokenizer.p", "rb") as f:
    tokenizer = pickle.load(f)

# Load the model
model = load_model('models/model_9.h5')

max_length = 32

@app.route('/')
def index():
    return render_template('index.html', caption='')

@app.route('/upload', methods=['POST'])
def upload():
    # Check if a file was uploaded
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    # Check if the file is empty
    if file.filename == '':
        return redirect(request.url)
    # Check if the file is an image
    if file and allowed_file(file.filename):
        # Process the image
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        photo = extract_features(img, xception_model)
        description = generate_desc(model, tokenizer, photo, max_length)
        return description
    else:
        return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)
