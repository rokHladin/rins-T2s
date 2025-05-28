import tkinter as tk
from tkinter.scrolledtext import ScrolledText
import threading
import queue
import spacy
import pyttsx3
import speech_recognition as sr
from enum import Enum
from typing import Optional
import pyaudio
import traceback
import time


BIRD_SPECIES = [
    "Laysan Albatross", "Yellow headed Blackbird", "Indigo Bunting", "Pelagic Cormorant",
    "American Crow", "Yellow billed Cuckoo", "Purple Finch", "Vermilion Flycatcher",
    "European Goldfinch", "Eared Grebe", "California Gull", "Ruby throated Hummingbird",
    "Blue Jay", "Pied Kingfisher", "Baltimore Oriole", "White Pelican", "Horned Puffin",
    "White necked Raven", "Great Grey Shrike", "House Sparrow", "Cape Glossy Starling",
    "Tree Swallow", "Common Tern", "Red headed Woodpecker"
]
BIRD_KEYWORDS = {
    "albatross": "Laysan Albatross", "blackbird": "Yellow headed Blackbird", "bunting": "Indigo Bunting",
    "cormorant": "Pelagic Cormorant", "crow": "American Crow", "cuckoo": "Yellow billed Cuckoo",
    "finch": "Purple Finch", "flycatcher": "Vermilion Flycatcher", "goldfinch": "European Goldfinch",
    "grebe": "Eared Grebe", "gull": "California Gull", "hummingbird": "Ruby throated Hummingbird",
    "jay": "Blue Jay", "kingfisher": "Pied Kingfisher", "oriole": "Baltimore Oriole",
    "pelican": "White Pelican", "puffin": "Horned Puffin", "raven": "White necked Raven",
    "shrike": "Great Grey Shrike", "sparrow": "House Sparrow", "starling": "Cape Glossy Starling",
    "swallow": "Tree Swallow", "tern": "Common Tern", "woodpecker": "Red headed Woodpecker",
    "robin": "House Sparrow", "hawk": "White necked Raven", "eagle": "White necked Raven", "stork": "White Pelican"
}
POSITIVE_KEYWORDS = ['yes', 'yeah', 'yep', 'sure', 'definitely', 'absolutely', 'affirmative', 'correct', 'right']
NEGATIVE_KEYWORDS = ['no', 'nope', 'nah', 'negative', 'not', 'never', 'absolutely not', 'wrong']

class Gender(Enum):
    WOMAN = "woman"
    MAN = "man"

class ChatboxGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Bird Dialogue Chat")
        self.text_area = ScrolledText(self.root, state='disabled', wrap='word', width=70, height=22, font=('Consolas', 11))
        self.text_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        self.bottom_frame = tk.Frame(self.root)
        self.bottom_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=(0,10))

        self.input_var = tk.StringVar()
        self.entry = tk.Entry(self.bottom_frame, textvariable=self.input_var, font=('Consolas', 12))
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.entry.bind('<Return>', self.on_send)
        self.send_button = tk.Button(self.bottom_frame, text="Send", command=self.on_send, font=('Consolas', 11))
        self.send_button.pack(side=tk.LEFT, padx=(5, 0))
        self.mic_button = tk.Button(self.bottom_frame, text="MIC", command=self.on_speech, font=('Consolas', 11))
        self.mic_button.pack(side=tk.LEFT, padx=(5, 0))
        self.exit_button = tk.Button(self.bottom_frame, text="Exit", command=self.on_exit, font=('Consolas', 11))
        self.exit_button.pack(side=tk.LEFT, padx=(10, 0))

        self._user_input = None
        self._input_ready = threading.Event()
        self._speech_ready = threading.Event()
        self._speech_input = None
        self.selected_device_index = None
        self._exiting = False  # <--- Added for clean exit

    def display(self, text):
        if self._exiting:
            return
        self.text_area.config(state='normal')
        self.text_area.insert(tk.END, text + "\n")
        self.text_area.config(state='disabled')
        self.text_area.see(tk.END)

    def get_user_input_blocking(self, prompt=""):
        if self._exiting:
            return None
        if prompt:
            self.display(prompt)
        self.entry.config(state='normal')
        self.entry.focus()
        self._input_ready.clear()
        while not self._input_ready.wait(timeout=0.1):
            if self._exiting:
                return None
            self.root.update()
        if self._exiting:
            return None
        result = self._user_input
        self._user_input = None
        self.entry.delete(0, tk.END)
        return result

    def get_user_input(self, prompt="", allow_voice=True):
        if self._exiting:
            return None
        if prompt:
            self.display(prompt)
        self._input_ready.clear()
        self._speech_ready.clear()
        self.entry.config(state='normal')
        self.entry.focus()
        while True:
            if self._exiting:
                return None
            if self._input_ready.wait(timeout=0.1):
                break
            if allow_voice and self._speech_ready.is_set():
                break
        if self._exiting:
            return None
        if self._input_ready.is_set():
            result = self._user_input
            self._user_input = None
            self.entry.delete(0, tk.END)
            return result
        elif self._speech_ready.is_set():
            return self._speech_input

    def on_send(self, event=None):
        if self._exiting:
            return
        self._user_input = self.input_var.get()
        if self._user_input.strip():
            self.display(f"You: {self._user_input}")
            self._input_ready.set()
        self.input_var.set("")

    def on_speech(self):
        if self._exiting:
            return
        self._speech_ready.clear()
        def recognize_in_thread():
            r = sr.Recognizer()
            try:
                with sr.Microphone(device_index=self.selected_device_index) as source:
                    self.display("[Microphone] Listening...")
                    r.adjust_for_ambient_noise(source, duration=1)
                    audio = r.listen(source, timeout=10)
                text = r.recognize_google(audio)
                self._speech_input = text
                self.display(f"You (mic): {text}")
            except sr.UnknownValueError:
                self._speech_input = ""
                self.display("[Mic error] Sorry, I couldn't understand your speech. Please try again, or type your answer.")
            except Exception as e:
                self._speech_input = ""
                tb = traceback.format_exc()
                self.display(f"[Mic error] {e}\n{tb}")
            self._speech_ready.set()
        threading.Thread(target=recognize_in_thread, daemon=True).start()

    def on_exit(self):
        self._exiting = True
        self._input_ready.set()
        self._speech_ready.set()
        self.root.quit()
        self.root.destroy()

    def start(self):
        self.root.mainloop()

class SpeechResult:
    def __init__(self, text: str, nlp):
        self.text = text
        self.doc = nlp(text) if nlp else None

    def extract_bird(self) -> Optional[str]:
        text_lower = self.text.lower()
        for keyword, bird_name in BIRD_KEYWORDS.items():
            if keyword in text_lower:
                return bird_name
        for bird in BIRD_SPECIES:
            bird_words = bird.lower().split()
            for word in bird_words:
                if len(word) > 3 and word in text_lower:
                    return bird
        return None

    def is_positive_response(self) -> bool:
        text_lower = self.text.lower()
        if any(word in text_lower for word in NEGATIVE_KEYWORDS):
            return False
        if any(word in text_lower for word in POSITIVE_KEYWORDS):
            return True
        return False

    def is_negative_response(self) -> bool:
        text_lower = self.text.lower()
        return any(word in text_lower for word in NEGATIVE_KEYWORDS)

    def get_text(self) -> str:
        return self.text

class SpeechGUI:
    def __init__(self, gui: ChatboxGUI, enable_tts=True):
        self.gui = gui
        self.enable_tts = enable_tts
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 150)
            self.engine.setProperty('volume', 0.9)
        except Exception:
            self.engine = None
            self.enable_tts = False
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            self.nlp = None
            self.gui.display("Warning: spaCy model not found. Install with: python -m spacy download en_core_web_sm")

    def recognize(self) -> SpeechResult:
        user_input = self.gui.get_user_input("[Your answer] (Type or use MIC button): \n", allow_voice=True)
        return SpeechResult(user_input or "", self.nlp)

    def speak(self, text: str, wait=False):
        self.gui.display(f"Robot: {text}")
        if self.enable_tts and self.engine:
            def tts_job():
                try:
                    self.engine.say(text + " .")
                    self.engine.runAndWait()
                    time.sleep(0.8)
                except Exception:
                    pass

            if wait:
                tts_job()
            else:
                t = threading.Thread(target=tts_job, daemon=True)
                t.start()


class BirdDialogue:
    def __init__(self, speech_system):
        self.speech = speech_system
        self.bird_locations = {
            "Laysan Albatross": "in the north part of the park sitting on a blue ring",
            "Yellow headed Blackbird": "in the east part of the park sitting on a yellow ring",
            "Indigo Bunting": "in the center of the park sitting on a green ring",
            "Pelagic Cormorant": "in the west part of the park sitting on a red ring",
            "American Crow": "in the south part of the park sitting on a black ring",
            "Yellow billed Cuckoo": "in the east part of the park sitting on a blue ring",
            "Purple Finch": "in the center of the park sitting on a purple ring",
            "Vermilion Flycatcher": "in the west part of the park sitting on a red ring",
            "European Goldfinch": "in the north part of the park sitting on a yellow ring",
            "Eared Grebe": "in the south part of the park sitting on a green ring",
            "California Gull": "in the center of the park sitting on a white ring",
            "Ruby throated Hummingbird": "in the east part of the park sitting on a red ring",
            "Blue Jay": "in the north part of the park sitting on a blue ring",
            "Pied Kingfisher": "in the west part of the park sitting on a green ring",
            "Baltimore Oriole": "in the south part of the park sitting on an orange ring",
            "White Pelican": "in the center of the park sitting on a white ring",
            "Horned Puffin": "in the north part of the park sitting on a black ring",
            "White necked Raven": "in the east part of the park sitting on a black ring",
            "Great Grey Shrike": "in the west part of the park sitting on a gray ring",
            "House Sparrow": "in the center of the park sitting on a brown ring",
            "Cape Glossy Starling": "in the south part of the park sitting on a purple ring",
            "Tree Swallow": "in the west part of the park sitting on a red ring",
            "Common Tern": "in the east part of the park sitting on a white ring",
            "Red headed Woodpecker": "in the north part of the park sitting on a red ring"
        }

    def get_bird_location(self, bird_name: str) -> str:
        return self.bird_locations.get(bird_name, "somewhere in the park")

    def conduct_dialogue(self, gender: Gender) -> Optional[str]:
        if gender == Gender.WOMAN:
            return self._dialogue_with_woman()
        else:
            return self._dialogue_with_man()

    def _dialogue_with_woman(self) -> Optional[str]:
        max_attempts = 5
        attempts = 0
        while attempts < max_attempts:
            if self.speech.gui._exiting:
                return None
            if attempts == 0:
                self.speech.speak("Hi woman, which is your favourite bird?")
            else:
                self.speech.speak("I'm sorry, I didn't recognize that bird. I will ask again. Which is your favourite bird?")
            response = self.speech.recognize()
            if self.speech.gui._exiting:
                return None
            bird = response.extract_bird()
            if bird:
                location = self.get_bird_location(bird)
                self.speech.speak(f"Well there is one {bird.lower()} {location}.")
                return bird
            attempts += 1
        self.speech.speak("I'm sorry, I still couldn't determine your favourite bird.")
        return None

    def _dialogue_with_man(self) -> Optional[str]:
        max_attempts = 5
        attempts = 0
        while attempts < max_attempts:
            if self.speech.gui._exiting:
                return None
            if attempts == 0:
                self.speech.speak("Hi man, which is your favourite bird?")
            else:
                self.speech.speak("I'm sorry, I didn't recognize that bird. I will ask again. Which is your favourite bird?")
            response = self.speech.recognize()
            if self.speech.gui._exiting:
                return None
            bird = response.extract_bird()
            if not bird:
                attempts += 1
                continue
            # Confirmation section
            bird_mentions = [bird]
            confirm_attempts = 5
            confirm_count = 0
            last_prompt = None
            while confirm_count < confirm_attempts:
                if self.speech.gui._exiting:
                    return None
                # Only repeat "Are you sure?" if it wasn't just asked
                if last_prompt != "are_you_sure":
                    self.speech.speak("Are you sure?")
                    last_prompt = "are_you_sure"
                else:
                    self.speech.speak("Please confirm your choice.")
                    last_prompt = "confirm_choice"
                confirmation = self.speech.recognize()
                if self.speech.gui._exiting:
                    return None
                if confirmation.is_positive_response():
                    location = self.get_bird_location(bird)
                    self.speech.speak(f"There is one {bird.lower()} {location}.")
                    return bird
                new_bird = confirmation.extract_bird()
                if new_bird:
                    bird_mentions.append(new_bird)
                    bird = new_bird
                    # Reset prompt logic after new bird
                    last_prompt = None
                    if bird_mentions.count(new_bird) >= 2:
                        location = self.get_bird_location(new_bird)
                        self.speech.speak(f"OK. The {new_bird.lower()} then. There is one {location}.")
                        return new_bird
                    self.speech.speak(f"OK, the {bird.lower()} then.")
                confirm_count += 1
            if bird:
                location = self.get_bird_location(bird)
                self.speech.speak(f"OK. The {bird.lower()} then. There is one {location}.")
                return bird
            attempts += 1
        self.speech.speak("I'm sorry, I still couldn't determine your favourite bird.")
        return None


def run_bird_dialogue_gui(gender: Gender, rings, birds, tts=True):
    gui = ChatboxGUI()

    #debug display birds rings arrays
    #gui.display("\nTESTING SECTION\n")
    #gui.display(f"\n{birds}\n")
    #gui.display(f"\n{rings}\n")


    # List input devices in GUI, let user pick synchronously in main thread
    p = pyaudio.PyAudio()
    devices_info = [p.get_device_info_by_index(i) for i in range(p.get_device_count())]
    gui.display("Available input devices:\n")
    input_indexes = []
    for idx, dev in enumerate(devices_info):
        if dev.get('maxInputChannels', 0) > 0:
            gui.display(f"{idx}: {dev.get('name')} (inputs: {dev.get('maxInputChannels',0)})")
            input_indexes.append(idx)
    gui.display("\nPlease type the device index to use for the microphone (e.g., 0):")
    while True:
        idx = gui.get_user_input_blocking("[Select device index]:")
        if gui._exiting:
            return
        try:
            idx = int(idx)
            if idx in input_indexes:
                gui.selected_device_index = devices_info[idx]['index']
                gui.display(f"Selected input device: {devices_info[idx]['name']}\n")
                break
            else:
                gui.display("Invalid selection. Please pick an index with inputs > 0.")
        except Exception:
            gui.display("Invalid input. Please enter a number.")

    speech_system = SpeechGUI(gui, enable_tts=tts)
    dialogue = BirdDialogue(speech_system)
    gui.display(f"\n=== Starting dialogue with {gender.value} ===\n")

    def run_dialogue():
        favorite_bird = dialogue.conduct_dialogue(gender)
        if gui._exiting:
            return
        if favorite_bird:
            result_line = f"Hope that helps"
            speech_system.speak(result_line, wait=True)
        else:
            result_line = f"I'm sorry I could not find your bird. Sad Sad"
            speech_system.speak(result_line, wait=True)

    gui.display("\n(You may close this window when done.)\n")
    threading.Thread(target=run_dialogue, daemon=True).start()
    gui.start()

if __name__ == "__main__":
    run_bird_dialogue_gui(Gender.MAN, [], [])
