import logging
from pdf2image import convert_from_path
from PIL import Image
import requests
import os
import json
import openai
from moviepy.editor import ImageClip, concatenate_videoclips, AudioFileClip
from docs_converter import *

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def clean_files_in_directory(directory):
    # Loop through all files and directories in the specified directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)  # Create full path
        # Check if it is a file
        if os.path.isfile(file_path):
            os.remove(file_path)  # Remove the file
            print(f"Deleted {file_path}")
        else:
            print(f"Skipping {file_path} (not a file)")


# class PDFToJPEGConverter:
#     def __init__(self, pdf_file_path, images_dir):
#         self.pdf_file_path = pdf_file_path
#         self.images_dir = images_dir
#         # Set up logging

#         # Create the images directory if it doesn't exist
#         if not os.path.exists(self.images_dir):
#             os.makedirs(self.images_dir)
#         else:
#             # Clean the directory if it already exists
#             clean_files_in_directory(self.images_dir)

#     def convert(self):
#         """
#         Convert each page of the PDF to a separate JPEG file.
#         """
#         try:
#             # Convert PDF to a list of images
#             images = convert_from_path(self.pdf_file_path)
#             logging.info(
#                 f"Successfully converted PDF to images: {self.pdf_file_path}")

#             # Save each image as JPEG
#             for index, image in enumerate(images):
#                 image_path = f"{self.images_dir}/page_{index + 1}.jpg"
#                 image.save(image_path, 'JPEG')
#                 logging.info(f"Saved JPEG file at {image_path}")
#         except FileNotFoundError:
#             logging.error(
#                 "The specified PDF file was not found. Please check the file path.")
#         except Exception as e:
#             logging.error(f"An error occurred during conversion: {str(e)}")


class ImageProcessorClient:
    def __init__(self, api_url, images_dir, scripts_dir):
        self.api_url = api_url
        self.images_dir = images_dir
        self.scripts_dir = scripts_dir
        self.context = []
        self.ai_client = OpenAIAPI()
        self.last_page_script = None

        # Create the response directory if it doesn't exist
        if not os.path.exists(self.scripts_dir):
            os.makedirs(self.scripts_dir)
        else:
            # Clean the directory if it already exists
            clean_files_in_directory(self.scripts_dir)

    # Function to extract the numerical part from the filename
    def _extract_number(self, filename):
        # Assuming the format is "page_X.ext"
        basename = os.path.splitext(filename)[0]  # Remove the extension
        number_part = basename.split('_')[1]      # Get the part after "page_"
        # Convert to integer for correct numerical sorting
        return int(number_part)

    def process_images(self, question):
        # List all image files in the directory
        filenames = [filename for filename in os.listdir(self.images_dir)
                     if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        filenames.sort(key=self._extract_number)
        page_number = 1
        for filename in filenames:
            file_path = os.path.join(self.images_dir, filename)
            # Process each image
            self.image_to_script(file_path, filename, question, page_number)
            page_number += 1

    def image_to_script(self, image_file_path, filename, question, page_number):
        logging.info(f"Processing image: {filename}")
        prompt = f"What is in the slide?"

        prompt2 = f"Please analyze the provided slide images, carefully extract all visible text, \
            and accurately transcribe it. Focus on maintaining the exact wording and punctuation as shown on the slides. \
                Ensure that the transcript is coherent, well-structured, and follows the logical order of information as \
                    presented on the slides. Additionally, identify and highlight any key points or headings to distinguish \
                        them from general content. The final output should be presented in a clear, readable format that can \
                            be used for a presentation."

        prompt3 = f"This is {page_number} page of the slide. \Please analyze the provided slide images and transcribe all visible\
              text. For the first slide, which typically serves as a welcome or introductory slide, ensure the transcription\
                  captures the welcoming nature and any introductory details succinctly. For content slides, maintain exact wording\
                    , punctuation, and logical order of the information. When encountering slides labeled as 'Thank You', \
                        'Questions', 'Sources', or 'References', provide a concise summary without delving into details. Highlight \
                            key points and headings to distinguish them from general content. The final transcript should be clear, \
                                structured, and suitable for presentation purposes, with special attention to the context of \
                                    each slide type."

        prompt4 = f"Please analyze the provided slide images to determine the type of each slide (e.g., 'Welcome', 'Content',\
              'Thank You', 'Questions', 'Q&A', 'Sources', 'References', 'Appendix') and accurately extract all visible text. \
                For 'Welcome' slides, focus on capturing any introductory or welcoming messages. For 'Content' slides, extract \
                    detailed information, maintaining the exact wording and punctuation. For 'Thank You', 'Questions', 'Q&A', \
                        'Sources', 'References', or 'Appendix' slides, summarize the key points succinctly. Label each extracted \
                            segment with its corresponding slide type for precise identification and further processing. generate \
                                in plain text but not markdown format. The final output should be clear, structured, and suitable "
        information = self.process_image(
            image_file_path, filename, prompt4)

        prompt5 = f"This is {page_number} page of the slide. Given the extracted information and the type of the current slide: \
            {information}, generate a coherent script for the speaker. If the current slide is a 'Welcome' slide, \
                begin with a warm introduction. For 'Content' slides, develop the information into a detailed narrative appropriate \
                    for presentation, building on themes or points introduced previously. For slides categorized as 'Thank You', \
                        'Questions', 'Q&A', 'Sources', 'References', or 'Appendix', briefly highlight the main points, linking them \
                            to the context provided by the previous slide where relevant. Ensure the transition between slides is \
                                smooth, maintaining a natural flow and keeping the audience engaged. Please remember this script will \
                                    be used for text to speech conversion, so just provide the speaker's content."

        prompt6 = f"This is page {page_number} of the slide. Using the extracted information and the type of the current slide: {information}, \
            generate a coherent speaking script for the speaker. Start directly with the greeting or main content without including \
                slide titles or formatting cues. For example, if the slide is a 'Welcome' slide, begin with a greeting like \
                    'Ladies and Gentlemen, I am thrilled to welcome you...'. For 'Content' slides, proceed directly into the \
                        discussion of the topic. Ensure the script flows naturally from one slide to the next, maintaining continuity \
                            and engagement, and omitting any preliminary or administrative details not meant for verbal presentation."

        prompt7 = f"This is page {page_number} of the slide. Using the extracted information and the type of the current slide: \
            {information}, generate a coherent speaking script for the speaker. For the first page, begin with a formal greeting \
                such as 'Ladies and Gentlemen, I am thrilled to welcome you...'. Ensure the script flows naturally from one slide \
                    to the next, maintaining continuity and engagement, and omitting any preliminary or administrative details not meant \
                        for verbal presentation. The script should be no more than 5 sentences long."

        prompt8 = f"Here is the extracted information of the current slide: \
            {information}. Use the extracted information to generate a coherent speaking script for the speaker. The script shall begin with\
                  'Moving on' or ‘Building on that point' or 'This bring us to'. Focus on delivering the topic clearly and \
                    engagingly. Ensure the script flows naturally from one slide \
                    to the next, maintaining continuity and engagement, and omitting any preliminary or administrative details not meant \
                        for verbal presentation. The script should be no more than 5 sentences long."
        logging.info(f"Extracted information: {information}")
        # script = self.process_image(
        #    image_file_path, filename, 'you are going to present this page to the audence, what would you say?')
        # logging.info(f"Original script: {script}")
        self.save_response('raw_' + filename, information)
        script = ""

        counter = 1
        while True:
            logging.info(f"Processing: {counter}")
            if page_number == 1:
                script = self.ai_client.call_openai_api(
                    'You are an experienced instructor who is writing scripts for your slides to present', prompt7)
            else:
                script = self.ai_client.call_openai_api(
                    f"You are an experienced instructor who is writing scripts for your slides to present", prompt8)
            logging.info(f"Script: {script}")
            editor_model_feedback = self.ai_client.call_openai_api(
                'You are an experience editor', f"Just answer Yes or No. Is the script good enough for the speaker? {script}")
            logging.info(f"Feedback: {editor_model_feedback}")
            if editor_model_feedback.strip().startswith('Yes'):
                break
            counter += 1
        self.last_page_script = script
        self.save_response(filename, script)

    def process_image(self, file_path, filename, question="What is in the image?"):
        logging.debug(f"file path: {file_path}")
        logging.debug(f"file name: {filename}")
        with open(file_path, 'rb') as image_file:
            files = {'image': image_file}
            data = {'question': question}
            # Send the image to the API
            response = requests.post(self.api_url, files=files, data=data)

            if response.status_code == 200:
                # Parse the response
                response_data = response.json()
                return response_data['response']
            else:
                print(
                    f'Error processing {filename}: {response.json().get("error")}')

    def save_response(self, filename, response):
        # Derive text file name from image file name
        text_file_path = os.path.join(
            self.scripts_dir, f"{os.path.splitext(filename)[0]}.txt")
        with open(text_file_path, 'w') as f:
            json.dump(response, f, indent=4)


class TextToSpeechClient:
    def __init__(self, api_url, txt_folder, wav_output_folder):
        self.api_url = api_url
        self.txt_folder = txt_folder
        self.wav_output_folder = wav_output_folder

        # Ensure the output folder exists
        if not os.path.exists(self.wav_output_folder):
            os.makedirs(self.wav_output_folder)
        else:
            # Clean the directory if it already exists
            clean_files_in_directory(self.wav_output_folder)

    def process_txts(self, speaker_wav, language):
        # List all files in the directory
        files = os.listdir(self.txt_folder)
        files.sort()
        for filename in files:
            if filename.lower().endswith('.txt') and filename.lower().startswith('page_'):
                file_path = os.path.join(self.txt_folder, filename)
                # Read the content of the text file
                with open(file_path, 'r', encoding='utf-8') as file:
                    text_content = file.read()
                    # Process the text content
                    self.text_to_speech(
                        text_content, speaker_wav, language, filename)

    def text_to_speech(self, text_content, speaker_wav, language, txt_filename):
        # Set up the payload for the POST request
        payload = {
            "text": text_content,
            "speaker_wav": speaker_wav,
            "language": language
        }

        # Send the POST request to the API
        response = requests.post(self.api_url, json=payload, stream=True)

        # Check if the request was successful
        if response.status_code == 200:
            self.save_audio(response, txt_filename)
        else:
            print(
                f"Failed to convert text to speech: {response.status_code} - {response.text}")

    def save_audio(self, response, filename):
        # Determine the path to save the file
        file_path = os.path.join(
            self.wav_output_folder, f"{os.path.splitext(filename)[0]}.wav")

        # Write the response content to a file
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Audio file saved as {file_path}")


class VideoCreator:
    def __init__(self, file_name_prefix, image_dir, wav_dir, num_pages, output_file='output.mp4', fps=24):
        """
        Initializes the VideoCreator class.

        :param num_pages: Number of pages (and corresponding number of image/audio pairs).
        :param output_file: Name of the output video file.
        """
        self.num_pages = num_pages
        self.output_file = output_file
        self.file_name_prefix = file_name_prefix
        self.image_dir = image_dir
        self.wav_dir = wav_dir
        self.fps = fps

    def create_video(self):
        """
        Creates a video by combining image and audio files into an MP4 video.
        """
        clips = []

        # Loop through each pair of image and audio file
        for i in range(1, self.num_pages + 1):
            # Construct filenames
            img_filename = f'{self.file_name_prefix}{i}.jpg'
            img_full_path = os.path.join(self.image_dir, img_filename)
            audio_filename = f'{self.file_name_prefix}{i}.wav'
            audio_full_path = os.path.join(self.wav_dir, audio_filename)

            # Load the image as a clip with duration of the audio
            audio_clip = AudioFileClip(audio_full_path)
            img_clip = ImageClip(img_full_path, duration=audio_clip.duration)

            # Set the audio of the image clip
            img_clip = img_clip.set_audio(audio_clip)

            # Append the clip to the list of clips
            clips.append(img_clip)

        # Concatenate all the clips together
        final_clip = concatenate_videoclips(clips, method="compose")

        # Write the result to a file
        final_clip.write_videofile(
            self.output_file, codec='mpeg4', fps=self.fps, preset='fast', audio_codec='aac', threads=4)


class OpenAIAPI:

    def __init__(self, base_url=None, model=None):
        self.CONST_BASE_URL = base_url or "http://gpu:5000/v1"
        self.CONST_MODEL = model or "llama3:8b-instruct-q4_K_M"

    def call_openai_api(self, system_prompt, user_prompt: str):
        client = openai.OpenAI(
            base_url=self.CONST_BASE_URL,
            api_key="Your-OpenAI-API-Key"
        )
        completion = client.chat.completions.create(
            model=self.CONST_MODEL,
            messages=[
                {"role": "system",
                    "content": system_prompt or "You are a helpful assistant."},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0,
        )

        if completion.choices:
            return completion.choices[0].message.content
        else:
            raise Exception(
                "Failed to receive a valid response from the OpenAI API")


def smoke_test_pdf_to_jpeg_converter():
    pdf_file_path = 'slides/Presentation.pptx'
    output_folder = 'images'
    converter = ConverterFactory.create_converter(pdf_file_path, output_folder)
    converter.convert()


def smoke_test_image_processor_client():
    api_url = "http://gpu:6000/process-image"
    images_dir = "images"
    response_dir = "txt"
    quetion = "You are a presentor of a tech talk. Explain to the audience what you want to present based on the image."
    client = ImageProcessorClient(api_url, images_dir, response_dir)
    client.process_images(question=quetion)


def smoke_test_text_to_speech_client():
    api_url = "http://gpu:8020/tts_to_audio/"
    output_folder = "wav_output/"
    txt_folder = "txt"
    client = TextToSpeechClient(api_url, txt_folder, output_folder)
    client.process_txts('female.wav', 'en')


def smoke_test_openai_api():
    openai_api = OpenAIAPI(
        'http://gpu:5000', '/data/dev/models/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf')
    response = openai_api.call_openai_api(
        "Hello world! How are you doing today?")
    print(response)


def smoke_test_video_creator():
    file_name_prefix = 'page_'
    image_dir = 'images'
    wav_dir = 'wav_output'
    num_pages = 16
    output_file = 'output.mp4'
    video_creator = VideoCreator(
        file_name_prefix, image_dir, wav_dir, num_pages, output_file, fps=24)
    video_creator.create_video()


def main():
    smoke_test_pdf_to_jpeg_converter()
    # smoke_test_image_processor_client()
    # smoke_test_text_to_speech_client()
    # smoke_test_video_creator()


if __name__ == "__main__":
    main()
