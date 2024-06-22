"""
Flask server for processing images and questions using a vision model.
This server provides endpoints for image processing and question answering
using a vision language model.
"""

import yaml
from abc import ABC, abstractmethod
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from io import BytesIO
import logging
import requests

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all domains on all routes


class IVisionLLM(ABC):
    """Abstract base class for vision language models."""

    @abstractmethod
    def chat(self, image, msgs, temperature):
        """Process an image and messages to generate a response."""
        pass


class MiniCPMV(IVisionLLM):
    """Implementation of the MiniCPM vision language model."""

    _instance = None
    model = None
    tokenizer = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MiniCPMV, cls).__new__(cls)
        return cls._instance

    def load_model_file(self, model_path, trust_remote_code=True):
        """Load the model and tokenizer from the specified path or use default."""
        try:
            MiniCPMV.model = AutoModel.from_pretrained(
                model_path or 'openbmb/MiniCPM-Llama3-V-2_5-int4',
                trust_remote_code=trust_remote_code
            )
            MiniCPMV.tokenizer = AutoTokenizer.from_pretrained(
                model_path or 'openbmb/MiniCPM-Llama3-V-2_5-int4',
                trust_remote_code=trust_remote_code
            )
            logger.info("Model and tokenizer loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model and tokenizer: {e}")
            raise

    def chat(self, image, msgs, temperature=0):
        """Process the image and messages using the model."""
        try:
            res = MiniCPMV.model.chat(
                image=image,
                msgs=msgs,
                tokenizer=MiniCPMV.tokenizer,
                sampling=True,
                temperature=temperature
            )
            logger.info("Model processed request successfully")
            return res
        except Exception as e:
            logger.error(f"Model processing failed: {e}")
            raise


@app.route('/v1/chat/completions', methods=['POST'])
def process_image_v1(vision_model: IVisionLLM = MiniCPMV()):
    """
    Endpoint for processing images and questions using the vision model.
    Expects a JSON payload with 'messages' containing image URL and question.
    """
    try:
        if 'messages' not in request.json:
            logger.error("No messages provided in request")
            return jsonify({'error': 'No messages provided'}), 400

        messages = request.json.get('messages', [])
        logger.info(f"Received {len(messages)} messages")

        image_url = next((content['image_url']['url']
                          for message in messages
                          if message['role'] == 'user'
                          for content in message['content']
                          if content['type'] == 'image_url'), None)

        if not image_url:
            logger.error("No image URL found in messages")
            return jsonify({'error': 'No image URL provided'}), 400

        logger.info(f"Processing image from URL: {image_url}")

        temperature = request.json.get('temperature', 0.7)
        logger.info(f"Using temperature: {temperature}")

        response = requests.get(image_url)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        image = Image.open(BytesIO(response.content)).convert('RGB')

        res = vision_model.chat(image, messages, temperature)
        return jsonify({'response': res}), 200

    except requests.RequestException as e:
        logger.error(f"Failed to fetch image: {e}")
        return jsonify({'error': 'Failed to fetch image'}), 400
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/process-image', methods=['POST'])
def process_image(vision_model: IVisionLLM = MiniCPMV()):
    """
    Endpoint for processing uploaded images and questions.
    Expects multipart/form-data with 'image' file and optional 'question'.
    """
    try:
        if 'image' not in request.files:
            logger.error("No image file in request")
            return jsonify({'error': 'No image provided'}), 400

        image_file = request.files['image']
        if not image_file:
            logger.error("Empty image file received")
            return jsonify({'error': 'Empty image file'}), 400

        logger.info(f"Processing image: {image_file.filename}")

        image = Image.open(BytesIO(image_file.read())).convert('RGB')
        question = request.form.get('question', 'What is in the image?')
        msgs = [{'role': 'user', 'content': question}]

        logger.info(f"Question: {question}")

        res = vision_model.chat(image, msgs, 0.7)
        return jsonify({'response': res}), 200

    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return jsonify({'error': 'Failed to process image'}), 500


def load_config_from_yaml(config_path):
    """Load configuration from a YAML file."""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logger.info("Config loaded successfully")
        return config
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        raise


if __name__ == '__main__':
    try:
        config = load_config_from_yaml('config.yaml')
        vision_model = MiniCPMV()
        vision_model.load_model_file(
            config.get('model_path'),
            config.get('trust_remote_code', True)
        )

        app.run(
            host=config.get('host', '0.0.0.0'),
            port=config.get('port', 5000)
        )
    except Exception as e:
        logger.critical(f"Failed to start the server: {e}")
