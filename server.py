# Description: A Flask server that processes images and questions using a vision model.
import yaml
from abc import ABC, abstractmethod
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from io import BytesIO
import logging
import requests

app = Flask(__name__)
CORS(app)  # Enable CORS for all domains on all routes

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class IVisionLLM(ABC):
    @abstractmethod
    def chat(self, image, msgs, temperature):
        pass


class MiniCPMV(IVisionLLM):
    _instance = None
    model = None
    tokenizer = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MiniCPMV, cls).__new__(cls)
            # Put any initialization here.
        return cls._instance

    def load_model_file(self, model_path, trust_remote_code=True):
        MiniCPMV.model = AutoModel.from_pretrained(
            model_path or 'openbmb/MiniCPM-Llama3-V-2_5-int4', trust_remote_code=trust_remote_code)
        MiniCPMV.tokenizer = AutoTokenizer.from_pretrained(
            model_path or 'openbmb/MiniCPM-Llama3-V-2_5-int4', trust_remote_code=trust_remote_code)

    def chat(self, image, msgs, temperature=0):
        try:
            # Process the image using the model
            res = MiniCPMV.model.chat(
                image=image,
                msgs=msgs,
                tokenizer=MiniCPMV.tokenizer,
                sampling=True,
                temperature=temperature
            )
            logging.info(f"Model returned a response {res}")
            return res
        except Exception as e:
            logging.error("Model processing failed: %s", e)
            return None


@app.route('/v1/chat/completions', methods=['POST'])
def process_image_v1(vision_model: IVisionLLM = MiniCPMV()):
    if 'messages' not in request.json:
        logging.error("No messages provided in request")
        return jsonify({'error': 'No messages provided'}), 400

    messages = request.json.get('messages', [])
    logging.info("Messages received: %s", messages)
    image_url = None

    # Extract image URL from messages
    for message in messages:
        if message['role'] == 'user':
            for content in message['content']:
                if content['type'] == 'image_url':
                    image_url = content['image_url']['url']
                    break
            if image_url:
                break
    temperature = request.json.get('temperature', 0.7)
    logging.info("Temperature for the model: %s", temperature)

    if not image_url:
        logging.error("No image URL found in messages")
        return jsonify({'error': 'No image URL provided'}), 400

    logging.info("Image URL received: %s", image_url)

    try:
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content)).convert('RGB')
        res = vision_model.chat(image, messages, 0.7)
        logging.info("Model returned a response")
        return jsonify({'response': res}), 200
    except Exception as e:
        logging.error("Model processing failed: %s", e)
        return jsonify({'error': 'Model processing error'}), 500


@app.route('/process-image', methods=['POST'])
def process_image(vision_model: IVisionLLM = MiniCPMV()):
    if 'image' not in request.files:
        logging.error("No image file in request")
        return jsonify({'error': 'No image provided'}), 400

    image_file = request.files['image']
    if image_file:
        logging.info("Image file received: %s", image_file.filename)

        try:
            # Convert the image file to an Image object
            image = Image.open(BytesIO(image_file.read())).convert('RGB')
            logging.debug("Image converted to RGB")
        except Exception as e:
            logging.error("Failed to process the image: %s", e)
            return jsonify({'error': 'Failed to process image'}), 422

        # Define the question or use one from the request
        question = request.form.get('question', 'What is in the image?')
        msgs = [{'role': 'user', 'content': question}]
        logging.debug("Question for the model: %s", question)

        try:
            res = vision_model.chat(image, msgs, 0.7)
            logging.info("Model returned a response")
            return jsonify({'response': res}), 200
        except Exception as e:
            logging.error("Model processing failed: %s", e)
            return jsonify({'error': 'Model processing error'}), 500

    logging.error("Invalid file format")
    return jsonify({'error': 'Invalid file'}), 400


def load_config_from_yaml(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


if __name__ == '__main__':
    config = load_config_from_yaml('config.yaml')
    logging.info("Config loaded successfully: %s", config)
    vision_model = MiniCPMV()
    # if model path is provided, load the model otherwise use default model
    vision_model.load_model_file(config.get(
        'model_path', None), config.get('trust_remote_code', True))
    # if host and port are provided in config, use them otherwise use default values
    app.run(host=config.get('host', '0.0.0.0'),
            port=config.get('port', 5000))
