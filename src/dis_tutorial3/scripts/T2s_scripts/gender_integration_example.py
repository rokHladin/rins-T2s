#!/usr/bin/env python3
"""
Example script showing how to integrate gender classification with dialogue system
"""

import rclpy
import time
from face_detector import FaceDetector
from dialogue import run_bird_dialogue, Gender

class GenderAwareDialogueNode:
    def __init__(self):
        # Initialize the face detector with gender classification
        self.face_detector = FaceDetector()
        
        # Wait a bit for the system to initialize
        time.sleep(2)
        
    def wait_for_person_and_dialogue(self, timeout=30):
        """Wait for a person to be detected and start appropriate dialogue"""
        print("Waiting for a person to be detected...")
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Get the most confident gender detection
            gender, confidence = self.face_detector.get_most_confident_gender()
            
            if gender and confidence > 0.7:  # Confidence threshold
                print(f"Detected {gender} with confidence {confidence:.2f}")
                
                # Convert to dialogue Gender enum
                dialogue_gender = Gender.WOMAN if gender == "woman" else Gender.MAN
                
                # Start the appropriate dialogue
                print(f"Starting dialogue with {dialogue_gender.value}...")
                favorite_bird = run_bird_dialogue(
                    gender=dialogue_gender,
                    use_keyboard=True,  # Use keyboard for testing
                    disable_tts=False
                )
                
                return favorite_bird
            
            time.sleep(0.1)  # Check 10 times per second
        
        print("Timeout: No person detected with sufficient confidence")
        return None

def main():
    rclpy.init()
    
    try:
        dialogue_node = GenderAwareDialogueNode()
        result = dialogue_node.wait_for_person_and_dialogue()
        
        if result:
            print(f"Successfully completed dialogue. Favorite bird: {result}")
        else:
            print("Failed to complete dialogue")
            
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        rclpy.shutdown()

if __name__ == "__main__":
    main() 