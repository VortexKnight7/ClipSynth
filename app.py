# ----------CODE Using My Own Model------------


from flask import Flask, render_template, request, send_from_directory
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import moviepy.editor as mp
import speech_recognition as sr
import pyttsx3
import os
# from refine import process_text_enhancement

app = Flask(__name__)

# Ensure the uploads folder exists
UPLOAD_FOLDER = 'uploads'
AUDIO_OUTPUT = os.path.join(UPLOAD_FOLDER, 'summary_audio.mp3')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Function for extracting audio from a video
def extract_audio(video_path, audio_output_path):
    try:
        video = mp.VideoFileClip(video_path)
        video.audio.write_audiofile(audio_output_path)
        print(f"Audio extracted and saved to {audio_output_path}")
    except Exception as e:
        print(f"Error in audio extraction: {e}")

# Function for transcribing audio
def transcribe_audio(audio_path):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
            transcript = recognizer.recognize_google(audio_data)
            print("Transcription completed successfully.")
            return transcript
    except Exception as e:
        print(f"Error in transcription: {e}")
        return None

# Function for generating audio from text
def text_to_audio_with_voice(text, output_audio_path, voice_index=1):
    try:
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        print("Available Voices:")
        for i, voice in enumerate(voices):
            print(f"Voice {i}: {voice.name}")

        if 0 <= voice_index < len(voices):
            engine.setProperty('voice', voices[voice_index].id)
            print(f"Using voice: {voices[voice_index].name}")
        else:
            print("Invalid voice index, using default voice.")

        engine.setProperty('rate', 120)
        print(f"Speech rate set to 120 words per minute.")

        engine.save_to_file(text, output_audio_path)
        engine.runAndWait()
        print(f"Audio saved to {output_audio_path}")
    except Exception as e:
        print(f"Error in text-to-audio conversion: {e}")

# Load the summarization model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("VortexKnight7/Video-Summ-Qwen")
model = AutoModelForCausalLM.from_pretrained("VortexKnight7/Video-Summ-Qwen").to(device)

@app.route('/', methods=['GET', 'POST'])
def Landing():
    return render_template('index.html')

@app.route('/transcript', methods=['GET', 'POST'])
def Transcript():
    return render_template('transcript_summarizer.html')

@app.route('/code_grn', methods=['GET', 'POST'])
def Code_Gen():
    return render_template('code_generator.html')

text = ""
new_summary = ""
@app.route('/summarize', methods=['POST'])
def summarize():
    input_type = request.form['input_type']

    if input_type == "link":
        video_link = request.form['video_link']
        unique_id = video_link.split("=")[-1]
        try:
            transcript = YouTubeTranscriptApi.get_transcript(unique_id)
            text = " ".join([x['text'] for x in transcript])
        except:
            return render_template('transcript_summarizer.html', error="Could not retrieve transcript.")

    elif input_type == "text":
        text = request.form['transcript_text']

    elif input_type == "file":
        file = request.files['transcript_file']
        if file and file.filename.endswith(('.txt', '.mp4', '.avi', '.mov', '.mkv', '.wav', '.mp3')):
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)
            if file.filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                audio_path = os.path.join(UPLOAD_FOLDER, 'extracted_audio.wav')
                extract_audio(file_path, audio_path)
                text = transcribe_audio(audio_path)
            elif file.filename.endswith(('.wav', '.mp3')):
                text = transcribe_audio(file_path)
            else:
                text = file.read().decode('utf-8')

    # Summarization
    prompt = f"""Query: Give me a brief summary on

Transcript:
{text}

Summary:
"""
    inputs = tokenizer([prompt], return_tensors="pt", padding=True, truncation=True).to(device)
    outputs = model.generate(**inputs, max_new_tokens=128, num_beams=1, early_stopping=True)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    new_summary = summary.split("Summary:")[-1].strip()

    # Generate audio for the summary
    text_to_audio_with_voice(new_summary, AUDIO_OUTPUT)
    # return new_summary
    return render_template('transcript_summarizer.html', summary=new_summary, audio_file='summary_audio.mp3')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# @app.route('/summarize1', methods=['POST'])
# def summarize1():
#     # ... [existing code remains the same] ...

#     # Use the new function for text enhancement
#     enhanced_summary = process_text_enhancement(new_summary)

#     # Generate audio for the enhanced summary
#     text_to_audio_with_voice(enhanced_summary, AUDIO_OUTPUT)

#     return render_template('trans.html', 
#                            summary=enhanced_summary, 
#                            audio_file='summary_audio.mp3')

if __name__ == '__main__':
    app.run(debug=True, port=5001)




# ----------CODE Using Facebook Bart------------

# from flask import Flask, render_template, request, send_from_directory
# from youtube_transcript_api import YouTubeTranscriptApi
# from transformers import BartTokenizer, BartForConditionalGeneration
# import torch
# import moviepy.editor as mp
# import speech_recognition as sr
# import pyttsx3
# import os

# app = Flask(__name__)

# # Ensure the uploads folder exists
# UPLOAD_FOLDER = 'uploads'
# AUDIO_OUTPUT = os.path.join(UPLOAD_FOLDER, 'summary_audio.mp3')
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# # Function for extracting audio from a video
# def extract_audio(video_path, audio_output_path):
#     try:
#         video = mp.VideoFileClip(video_path)
#         video.audio.write_audiofile(audio_output_path)
#         print(f"Audio extracted and saved to {audio_output_path}")
#     except Exception as e:
#         print(f"Error in audio extraction: {e}")

# # Function for transcribing audio
# def transcribe_audio(audio_path):
#     recognizer = sr.Recognizer()
#     try:
#         with sr.AudioFile(audio_path) as source:
#             audio_data = recognizer.record(source)
#             transcript = recognizer.recognize_google(audio_data)
#             print("Transcription completed successfully.")
#             return transcript
#     except Exception as e:
#         print(f"Error in transcription: {e}")
#         return None

# # Function for generating audio from text
# def text_to_audio_with_voice(text, output_audio_path, voice_index=1):
#     try:
#         engine = pyttsx3.init()
#         voices = engine.getProperty('voices')
#         print("Available Voices:")
#         for i, voice in enumerate(voices):
#             print(f"Voice {i}: {voice.name}")

#         if 0 <= voice_index < len(voices):
#             engine.setProperty('voice', voices[voice_index].id)
#             print(f"Using voice: {voices[voice_index].name}")
#         else:
#             print("Invalid voice index, using default voice.")

#         engine.setProperty('rate', 120)
#         print(f"Speech rate set to 120 words per minute.")

#         engine.save_to_file(text, output_audio_path)
#         engine.runAndWait()
#         print(f"Audio saved to {output_audio_path}")
#     except Exception as e:
#         print(f"Error in text-to-audio conversion: {e}")

# # Load the Facebook BART summarization model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
# model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn").to(device)

# @app.route('/', methods=['GET', 'POST'])
# def Landing():
#     return render_template('index.html')

# @app.route('/transcript', methods=['GET', 'POST'])
# def Transcript():
#     return render_template('transcript_summarizer.html')

# @app.route('/code_grn', methods=['GET', 'POST'])
# def Code_Gen():
#     return render_template('code_generator.html')

# @app.route('/summarize', methods=['POST'])
# def summarize():
#     input_type = request.form['input_type']
#     text = ""
#     new_summary = ""

#     if input_type == "link":
#         video_link = request.form['video_link']
#         unique_id = video_link.split("=")[-1]
#         try:
#             transcript = YouTubeTranscriptApi.get_transcript(unique_id)
#             text = " ".join([x['text'] for x in transcript])
#         except:
#             return render_template('transcript_summarizer.html', error="Could not retrieve transcript.")

#     elif input_type == "text":
#         text = request.form['transcript_text']

#     elif input_type == "file":
#         file = request.files['transcript_file']
#         if file and file.filename.endswith(('.txt', '.mp4', '.avi', '.mov', '.mkv', '.wav', '.mp3')):
#             file_path = os.path.join(UPLOAD_FOLDER, file.filename)
#             file.save(file_path)
#             if file.filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
#                 audio_path = os.path.join(UPLOAD_FOLDER, 'extracted_audio.wav')
#                 extract_audio(file_path, audio_path)
#                 text = transcribe_audio(audio_path)
#             elif file.filename.endswith(('.wav', '.mp3')):
#                 text = transcribe_audio(file_path)
#             else:
#                 text = file.read().decode('utf-8')

#     prompt = f"""Query: Summarize the following Transcripts:

# Transcript:
# {text}

# Summary:
# """

#     # Summarization with BART
#     inputs = tokenizer([prompt], return_tensors="pt", max_length=1024, truncation=True).to(device)
#     summary_ids = model.generate(inputs["input_ids"], max_length=200, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
#     new_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

#     # Generate audio for the summary
#     text_to_audio_with_voice(new_summary, AUDIO_OUTPUT)

#     return render_template('transcript_summarizer.html', summary=new_summary, audio_file='summary_audio.mp3')

# @app.route('/uploads/<filename>')
# def uploaded_file(filename):
#     return send_from_directory(UPLOAD_FOLDER, filename)

# if __name__ == '__main__':
#     app.run(debug=True, port=5001)
