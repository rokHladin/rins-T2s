import sys
import json
from T2s_custom_modules.dialogue import run_bird_dialogue_gui, Gender

if __name__ == '__main__':
    gender = Gender[sys.argv[1]]
    rings = json.loads(sys.argv[2])
    birds = json.loads(sys.argv[3])
    run_bird_dialogue_gui(gender, rings, birds, tts=True)
