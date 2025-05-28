import speech_recognition as sr
import pyttsx3
import spacy
import warnings
from enum import Enum
from typing import Optional

# Suppress pyttsx3 warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pyttsx3")

# Define bird species that can appear
BIRD_SPECIES = [
    "Laysan Albatross", "Yellow headed Blackbird", "Indigo Bunting", "Pelagic Cormorant",
    "American Crow", "Yellow billed Cuckoo", "Purple Finch", "Vermilion Flycatcher",
    "European Goldfinch", "Eared Grebe", "California Gull", "Ruby throated Hummingbird",
    "Blue Jay", "Pied Kingfisher", "Baltimore Oriole", "White Pelican", "Horned Puffin",
    "White necked Raven", "Great Grey Shrike", "House Sparrow", "Cape Glossy Starling",
    "Tree Swallow", "Common Tern", "Red headed Woodpecker"
]

# Create simplified bird name mapping for easier recognition
BIRD_KEYWORDS = {
    "albatross": "Laysan Albatross",
    "blackbird": "Yellow headed Blackbird",
    "bunting": "Indigo Bunting",
    "cormorant": "Pelagic Cormorant",
    "crow": "American Crow",
    "cuckoo": "Yellow billed Cuckoo",
    "finch": "Purple Finch",
    "flycatcher": "Vermilion Flycatcher",
    "goldfinch": "European Goldfinch",
    "grebe": "Eared Grebe",
    "gull": "California Gull",
    "hummingbird": "Ruby throated Hummingbird",
    "jay": "Blue Jay",
    "kingfisher": "Pied Kingfisher",
    "oriole": "Baltimore Oriole",
    "pelican": "White Pelican",
    "puffin": "Horned Puffin",
    "raven": "White necked Raven",
    "shrike": "Great Grey Shrike",
    "sparrow": "House Sparrow",
    "starling": "Cape Glossy Starling",
    "swallow": "Tree Swallow",
    "tern": "Common Tern",
    "woodpecker": "Red headed Woodpecker",
    # Additional common bird names for robustness
    "robin": "House Sparrow",  # Fallback
    "hawk": "White necked Raven",  # Fallback
    "eagle": "White necked Raven",  # Fallback
    "stork": "White Pelican"  # Fallback
}

# Keywords for positive/negative responses
POSITIVE_KEYWORDS = ['yes', 'yeah', 'yep', 'sure', 'definitely', 'absolutely', 'affirmative', 'correct', 'right']
NEGATIVE_KEYWORDS = ['no', 'nope', 'nah', 'negative', 'not', 'never', 'absolutely not', 'wrong']

class Gender(Enum):
    WOMAN = "woman"
    MAN = "man"

class SpeechResult:
    def __init__(self, text: str, nlp) -> None:
        self.text = text
        self.doc = nlp(text) if nlp else None
    
    def extract_bird(self) -> Optional[str]:
        """Extract bird name from the speech text"""
        text_lower = self.text.lower()
        
        # First try to find exact matches
        for keyword, bird_name in BIRD_KEYWORDS.items():
            if keyword in text_lower:
                return bird_name
        
        # If no exact match, try partial matching with bird species
        for bird in BIRD_SPECIES:
            # Split bird name into words and check if any are in the text
            bird_words = bird.lower().split()
            for word in bird_words:
                if len(word) > 3 and word in text_lower:  # Avoid short words
                    return bird
        
        return None
    
    def is_positive_response(self) -> bool:
        """Check if the response is positive (yes, sure, etc.)"""
        text_lower = self.text.lower()
        
        # Check for explicit negative words first
        if any(word in text_lower for word in NEGATIVE_KEYWORDS):
            return False
        
        # Check for positive words
        if any(word in text_lower for word in POSITIVE_KEYWORDS):
            return True
        
        return False
    
    def is_negative_response(self) -> bool:
        """Check if the response is negative"""
        text_lower = self.text.lower()
        return any(word in text_lower for word in NEGATIVE_KEYWORDS)
    
    def get_text(self) -> str:
        return self.text

class Speech:
    def __init__(self, nosr=False, disable_tts=False) -> None:
        self.r = sr.Recognizer()
        self.disable_tts = disable_tts
        self.engine = None
        
        # Initialize TTS engine with error handling
        if not disable_tts:
            try:
                self.engine = pyttsx3.init()
                # Set properties to reduce callback issues
                self.engine.setProperty('rate', 150)
                self.engine.setProperty('volume', 0.9)
            except Exception as e:
                print(f"[TTS INIT WARNING] Could not initialize TTS: {e}")
                self.disable_tts = True
        
        # Initialize spaCy
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Warning: spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # no speech recognition, keyboard only
        self.nosr = nosr
    
    def recognize_voice(self) -> SpeechResult:
        """Recognize speech from microphone"""
        with sr.Microphone() as source:
            self.r.adjust_for_ambient_noise(source, duration=1)
            print("[LISTENING] Speak now...")
            audio_data = self.r.listen(source, timeout=10)
            print("[PROCESSING] Recognizing...")
            try:
                text = self.r.recognize_google(audio_data)
                print(f"[HEARD] {text}")
                return SpeechResult(text, self.nlp)
            except sr.UnknownValueError:
                print("[ERROR] Could not understand the audio")
                raise
            except sr.RequestError:
                print("[ERROR] Could not request results; check your network connection")
                raise
    
    def recognize(self) -> SpeechResult:
        """Recognize speech with fallback to keyboard input"""
        if self.nosr:
            text = input("[KEYBOARD INPUT]: ")
            return SpeechResult(text, self.nlp)
        
        try:
            return self.recognize_voice()
        except Exception as e:
            print(f"[FALLBACK] Speech recognition failed: {e}")
            text = input("[KEYBOARD INPUT]: ")
            return SpeechResult(text, self.nlp)
    
    def speak(self, text: str):
        """Make the robot speak"""
        print(f"[ROBOT] {text}")
        
        if self.disable_tts or not self.engine:
            return
        
        try:
            # Create a new engine instance for each speech to avoid callback issues
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.engine.say(text)
                self.engine.runAndWait()
        except Exception as e:
            # Silently handle TTS errors to avoid cluttering output
            pass
    
    def __del__(self):
        """Cleanup TTS engine"""
        try:
            if self.engine:
                self.engine.stop()
        except:
            pass


class BirdDialogue:
    def __init__(self, speech_system: Speech):
        self.speech = speech_system
        # Mock bird location database
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
        """Get the location description for a bird"""
        return self.bird_locations.get(bird_name, "somewhere in the park")
    
    def conduct_dialogue(self, gender: Gender) -> Optional[str]:
        """Conduct the full dialogue based on gender"""
        if gender == Gender.WOMAN:
            return self._dialogue_with_woman()
        else:
            return self._dialogue_with_man()
    
    def _dialogue_with_woman(self) -> Optional[str]:
        """Handle dialogue with a woman (direct answer expected)"""
        self.speech.speak("Hi woman, which is your favourite bird?")
        
        response = self.speech.recognize()
        bird = response.extract_bird()
        
        if bird:
            location = self.get_bird_location(bird)
            self.speech.speak(f"Well there is one {bird.lower()} {location}.")
            return bird
        else:
            self.speech.speak("I'm sorry, I didn't recognize that bird. Could you repeat?")
            return None
    
    def _dialogue_with_man(self) -> Optional[str]:
        """Handle dialogue with a man (needs confirmation, may change mind)"""
        self.speech.speak("Hi man, which is your favourite bird?")
        
        response = self.speech.recognize()
        bird = response.extract_bird()
        
        if not bird:
            self.speech.speak("I'm sorry, I didn't recognize that bird. Could you repeat?")
            return None
        
        bird_mentions = [bird]  # Track mentioned birds
        max_attempts = 5  # Prevent infinite loops
        attempts = 0
        
        while attempts < max_attempts:
            attempts += 1
            
            # Ask for confirmation
            self.speech.speak("Are you sure?")
            confirmation = self.speech.recognize()
            
            if confirmation.is_positive_response():
                # Confirmed, we have the final answer
                location = self.get_bird_location(bird)
                self.speech.speak(f"There is one {bird.lower()} {location}.")
                return bird
            
            # Not confirmed, check if they mention a new bird
            new_bird = confirmation.extract_bird()
            if new_bird:
                bird_mentions.append(new_bird)
                bird = new_bird
                
                # Check if any bird has been mentioned twice
                for mentioned_bird in bird_mentions:
                    if bird_mentions.count(mentioned_bird) >= 2:
                        location = self.get_bird_location(mentioned_bird)
                        self.speech.speak(f"OK. The {mentioned_bird.lower()} then. There is one {location}.")
                        return mentioned_bird
                
                # Continue with the new bird
                self.speech.speak(f"OK, the {bird.lower()} then. Are you sure?")
            else:
                # No new bird mentioned, ask again
                self.speech.speak("Are you sure now?")
        
        # Fallback if max attempts reached
        if bird:
            location = self.get_bird_location(bird)
            self.speech.speak(f"OK. The {bird.lower()} then. There is one {location}.")
            return bird
        
        return None
    
# Main function to run the dialogue system
def run_bird_dialogue(gender: Gender, use_keyboard: bool = False, disable_tts: bool = False) -> Optional[str]:
    """
    Run the bird dialogue system
    
    Args:
        gender: Gender.WOMAN or Gender.MAN
        use_keyboard: If True, use keyboard input instead of microphone
        disable_tts: If True, disable text-to-speech output
    
    Returns:
        The favorite bird name if successfully determined, None otherwise
    """
    speech_system = Speech(nosr=use_keyboard, disable_tts=disable_tts)
    dialogue = BirdDialogue(speech_system)
    
    print(f"\n=== Starting dialogue with {gender.value} ===")
    print(f"Input method: {'Keyboard' if use_keyboard else 'Microphone'}")
    print(f"TTS: {'Disabled' if disable_tts else 'Enabled'}")
    print("="*50)
    
    favorite_bird = dialogue.conduct_dialogue(gender)
    
    print("="*50)
    if favorite_bird:
        print(f"[RESULT] Favorite bird determined: {favorite_bird}")
    else:
        print(f"[RESULT] Could not determine favorite bird")
    
    return favorite_bird

if __name__ == "__main__":
    # Example 1: Dialogue with a woman (keyboard input, TTS disabled to avoid callback errors)
    print("Example: Dialogue with a woman using keyboard input")
    print("Expected: Woman will give direct answer")

    # Uncomment the line below to run:
    result = run_bird_dialogue(Gender.WOMAN, use_keyboard=True, disable_tts=False)
    print(f"Result: {result}\n")