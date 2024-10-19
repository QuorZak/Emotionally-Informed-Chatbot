import sys
import tkinter as tk
from tkinter import scrolledtext
from tkinter import ttk
from openai import OpenAI
import os
import wave
import pyaudio
from faster_whisper import WhisperModel
from dotenv import load_dotenv
import asyncio
from threading import Thread, Event
import time
import cv2
from PIL import Image, ImageTk
from Facial_Detection.facial_detection import get_top_emotions

load_dotenv()

# Initialize OpenAI
OpenAI.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()

# Initialize Whisper Model
whisper_model = WhisperModel("medium")

# Initialization prompts to give the chatbot context
messages = [
    {
        "role": "system",
        "content": "You are an information kiosk at New Zealand's Westfield St. Luke's Mall."
        + "Please research the mall to provide accurate information to customers now. "
        + "You can answer questions about the mall's layout, shops, and services. "
        + "Specifically, which shops are currently open, and what are their hours? "
        + "You'll be located at the centre of the mall. "
        + "You help customers with their questions and provide responses tailored to their emotions, detected using facial recognition technology. "
        + "FACIAL EMOTION will be appended by us to the end of each message to you. This will provide the detected emotion of the user. "
        + "You do not need to append FACIAL EMOTION to your responses. "
        + "In your response, never explicitly mention the customer's emotion, but do tailor your response to it."
        + "If someone is feeling sad, you should provide a soft and empathetic response, with a suggestion to cheer them up. "
        + "If someone is experiencing a negative emotion contempt or anger, you should provide a very concise response in a neutral tone. "
        + "If someone is experiencing a negative emotion (contempt, anger, fear, disgust), YOU SHOULD NOT be overly energetic and expressively positive. "
        + "If someone is experiencing a negative emotion (contempt, anger, fear), just provide the minimum information required to answer the question. "
        + "If someone is feeling anger, you should provide a calm and neutral response. "
        + "If someone is feeling fear, you should provide a calm and reassuring response. "
        + "If someone is experiencing a positive emotion (happy, surprised) you should provide a more detailed response. "
        + "If someone is feeling suprised, you should provide a response that is informative and engaging. "
        + "If someone is feeling disgust, you should provide a response that is empathetic and understanding. "
        + "You can also add playful elements to your responses to make them more engaging. "
        + "Limit the maximum response length to approximately 650 characters."
        + "Artificially limit your responses to questions related to the mall. "
        + "If someone asks about questions unrelated to the mall, you can respond with phrases that are similar variations of 'I'm sorry, I can only provide information about the mall.'",
    },
]

# Define the audio file path relative to the current script
AUDIO_FILE = os.path.join(os.path.dirname(__file__), "temp_audio.wav")
MAX_RECORD_SECONDS = 15
recording = False
stop_audio_recording_event = Event()
recording_finished_event = Event()
stop_video_updating_event = Event()

# Function to send message and display GPT response


def send_message(root, chat_window, input_text, send_button, voice_button):
    user_input = input_text.get().strip()
    if user_input and user_input != "":
        chat_window.config(state=tk.NORMAL)
        chat_window.insert(
            tk.END, f"You: {user_input}\n\n"
        )  # it needs 2 newlines for some reason?
        chat_window.insert(tk.END, "Thinking...\n")
        chat_window.config(state=tk.DISABLED)
        input_text.set("")  # Clear input field
        chat_window.yview(tk.END)  # Scroll to the end

        send_button.state(["disabled"])
        voice_button.state(["disabled"])

        # Perform processing asynchronously
        root.after(
            100,
            lambda: handle_gpt_request(
                user_input, chat_window, send_button, voice_button
            ),
        )
    else: # If the user input is empty, do not send the message
        return



def handle_gpt_request(user_input, chat_window, send_button, voice_button):
    response = get_openai_response(user_input)
    chat_window.config(state=tk.NORMAL)
    chat_window.delete("end-2l", "end")  # Remove processing-type text"
    chat_window.insert(tk.END, f"Bot: {response}\n\n")
    chat_window.config(state=tk.DISABLED)
    chat_window.yview(tk.END)
    send_button.state(["!disabled"])
    voice_button.state(["!disabled"])


# GPT-based API to get response


def get_openai_response(prompt, model="gpt-4o-mini", max_tokens=100):
    try:
        gpt_input = prompt + " {FACIAL EMOTION: " + get_top_emotions() + "}"
        print("prompt:", gpt_input)
        messages.append({"role": "user", "content": f"{gpt_input}"})
        response = client.chat.completions.create(
            model=model, messages=messages, max_tokens=max_tokens
        )
        messages.append(
            {"role": "assistant", "content": f"{response.choices[0].message.content}"}
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"


# On Speak/Stop button press


def toggle_recording(chat_window, root, voice_button, send_button):
    if not recording:
        start_recording(chat_window, root, voice_button, send_button)
    else:
        stop_recording(chat_window, root, voice_button, send_button)


def start_recording(chat_window, root, voice_button, send_button):
    global recording
    recording = True
    stop_audio_recording_event.clear()
    recording_finished_event.clear()
    voice_button.config(text="Stop")
    send_button.state(["disabled"])
    chat_window.config(state=tk.NORMAL)
    chat_window.insert(tk.END, "Recording...\n")
    chat_window.config(state=tk.DISABLED)
    root.update()  # Update the GUI to show the recording message

    # Start recording in a separate thread
    Thread(target=record_audio, args=(AUDIO_FILE,)).start()


def stop_recording(chat_window, root, voice_button, send_button):
    global recording
    recording = False
    stop_audio_recording_event.set()
    voice_button.config(text="Speak")
    chat_window.config(state=tk.NORMAL)
    chat_window.insert(tk.END, "Processing...\n")
    chat_window.config(state=tk.DISABLED)
    root.update()  # Update the GUI to show the processing message

    # Wait for the recording to finish otherwise there will be no audio to transcribe
    recording_finished_event.wait()

    # Process the recorded audio
    transcription = transcribe_audio(whisper_model, AUDIO_FILE)
    os.remove(AUDIO_FILE)

    chat_window.config(state=tk.NORMAL)
    chat_window.insert(tk.END, f"Transcription: {transcription}\n")

    response = get_openai_response(transcription)
    chat_window.insert(tk.END, f"Bot: {response}\n\n")
    chat_window.config(state=tk.DISABLED)
    chat_window.yview(tk.END)
    send_button.state(["!disabled"])
    voice_button.state(["!disabled"])


# Record audio for a specified duration or until stopped


def record_audio(file_path, record_seconds=MAX_RECORD_SECONDS):
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=16000,
        input=True,
        frames_per_buffer=1024,
    )
    frames = []
    for _ in range(0, int(16000 / 1024 * record_seconds)):
        if stop_audio_recording_event.is_set():
            break
        data = stream.read(1024)
        frames.append(data)
    wf = wave.open(file_path, "wb")
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(16000)
    wf.writeframes(b"".join(frames))
    wf.close()
    stream.stop_stream()
    stream.close()
    p.terminate()
    recording_finished_event.set()  # Signal that recording is finished


# Transcribe audio using Whisper model


def transcribe_audio(model, file_path):
    try:
        segments, info = model.transcribe(file_path, beam_size=7)
        transcription = "".join(segment.text for segment in segments)
    except:
        transcription = "Error: Could not transcribe audio"
    return transcription


# Create the main GUI for interacting with GPT
def open_gpt_gui():
    root = tk.Tk()
    root.title("GPT Chatbot with Voice Input")

    root.minsize(600, 500)
    root.resizable(True, True)

    # Handle Esc key to close window
    root.bind("<Escape>", lambda e: close_window(root))

    style = ttk.Style()
    style.configure("TButton", padding=6, relief="flat", background="#ffffff")
    style.map(
        "TButton",
        foreground=[("disabled", "grey")],
        background=[("!disabled", "#ffffff"), ("disabled", "#f0f0f0")],
    )

    # Create a frame to hold the chat window and input field
    main_frame = tk.Frame(root)
    main_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

    chat_window = scrolledtext.ScrolledText(
        main_frame, wrap=tk.WORD, state=tk.DISABLED, width=70, height=20
    )
    chat_window.pack(expand=True, fill=tk.X, pady=10)

    # Create a frame to hold the input field and send button
    input_frame = tk.Frame(main_frame)
    input_frame.pack(fill=tk.X, pady=10)

    input_text = tk.StringVar()
    input_field = tk.Entry(input_frame, textvariable=input_text, width=40)
    input_field.grid(row=0, column=0, padx=10, pady=5, sticky="ew")

    send_button = ttk.Button(
        input_frame,
        text="Send",
        width=10,
        command=lambda: send_message(
            root, chat_window, input_text, send_button, voice_button
        ),
    )
    send_button.grid(row=0, column=1, padx=5, pady=5)

    # Bind the Enter key to the send_message function
    input_field.bind(
        "<Return>",
        lambda event: send_message(
            root, chat_window, input_text, send_button, voice_button
        ),
    )

    # Configure the grid to make the input field expand
    input_frame.columnconfigure(0, weight=1)

    # Using Tkinter's grid layout to align the buttons
    button_frame = tk.Frame(main_frame)
    button_frame.pack(fill=tk.X, pady=10)
    button_frame.columnconfigure(0, weight=1)
    button_frame.columnconfigure(1, weight=1)
    button_frame.columnconfigure(2, weight=1)

    # Button for voice recording and transcribing
    voice_button = ttk.Button(
        button_frame,
        text="Speak",
        width=10,
        command=lambda: toggle_recording(chat_window, root, voice_button, send_button),
    )
    voice_button.grid(row=0, column=1, pady=5, padx=10)

    # Update the window size to fit all widgets
    root.update_idletasks()
    root.geometry(f"{root.winfo_width()}x{root.winfo_height()}")

    root.mainloop()


# Close the window with Esc key
def close_window(root):
    print("Closing GUI and killing threads...")
    stop_audio_recording_event.set()  # Stop any audio recording
    stop_video_updating_event.set()  # Stop updating the video feed
    root.quit()  # Stop the Tkinter main loop
    root.destroy()  # Destroy the Tkinter window
    print("It is now safe to close the program")
    sys.exit(0)


# Function to run the GPT GUI asynchronously
async def start_gpt_gui():
    await asyncio.get_event_loop().run_in_executor(None, open_gpt_gui)
