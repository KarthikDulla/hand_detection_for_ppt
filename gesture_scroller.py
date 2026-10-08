import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# Screen dimensions for scrolling
screen_width, screen_height = pyautogui.size()

# Scrolling parameters
scroll_speed = 20
last_scroll_time = 0
scroll_cooldown = 0.1  # seconds between scrolls

# Gesture tracking
prev_y = None
gesture_threshold = 20  # minimum movement to register as gesture

def detect_scroll_gesture(hand_landmarks):
    """Detect vertical swipe gesture based on hand movement."""
    global prev_y, last_scroll_time

    # Get index finger tip position (landmark 8)
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    current_y = int(index_tip.y * screen_height)

    if prev_y is not None:
        # Calculate vertical movement
        dy = current_y - prev_y

        # Check if enough time has passed since last scroll
        current_time = time.time()
        if current_time - last_scroll_time > scroll_cooldown:
            if dy > gesture_threshold:  # Moving down -> scroll up
                pyautogui.scroll(scroll_speed)
                last_scroll_time = current_time
                return "SCROLL_UP"
            elif dy < -gesture_threshold:  # Moving up -> scroll down
                pyautogui.scroll(-scroll_speed)
                last_scroll_time = current_time
                return "SCROLL_DOWN"

    prev_y = current_y
    return "NO_GESTURE"

def draw_gesture_info(frame, gesture, fps):
    """Draw gesture information on the frame."""
    # Draw gesture text
    color = (0, 255, 0) if gesture != "NO_GESTURE" else (0, 0, 255)
    cv2.putText(frame, f"Gesture: {gesture}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Draw FPS
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # Draw instructions
    cv2.putText(frame, "Move index finger UP/DOWN to scroll", (10, screen_height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, "Press 'q' to quit", (10, screen_height - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

def main():
    """Main function to run the gesture scroller."""
    global prev_y

    # Initialize webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    print("Gesture Scroller Started!")
    print("Move your index finger up/down to scroll")
    print("Press 'q' to quit")

    prev_time = 0

    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break

        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process frame with MediaPipe
        results = hands.process(rgb_frame)

        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
        prev_time = current_time

        # Default gesture
        gesture = "NO_GESTURE"

        # Check if hand is detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Detect scroll gesture
                gesture = detect_scroll_gesture(hand_landmarks)

        # Reset prev_y if no hand detected (to prevent false gestures)
        if not results.multi_hand_landmarks:
            prev_y = None

        # Draw information on frame
        draw_gesture_info(frame, gesture, fps)

        # Display frame
        cv2.imshow('Gesture Scroller', frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("Gesture Scroller stopped!")

if __name__ == "__main__":
    main()