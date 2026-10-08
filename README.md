# Hand Detection for PowerPoint Control

A Python-based hand gesture recognition system that allows you to control PowerPoint presentations using simple hand movements. This project uses MediaPipe for hand tracking and pyautogui to simulate keyboard/mouse controls for navigating slides.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Usage](#usage)
- [Technical Details](#technical-details)
- [Future Enhancements](#future-enhancements)
- [License](#license)

## 🔍 Overview

This project enables hands-free control of PowerPoint presentations through real-time hand gesture recognition. By using your webcam and detecting specific hand movements, you can navigate through slides without needing a physical clicker or keyboard.

The system focuses on intuitive gestures:
- **Vertical finger movements** → Scroll up/down (useful for scrolling within slides or documents)
- **Future planned gestures** → Slide navigation (next/previous), pointer control, etc.

## ✨ Features

- **Real-time hand tracking** using MediaPipe
- **Gesture-based scrolling** (index finger up/down movement)
- **Low latency response** for smooth presentation control
- **Visual feedback** with on-screen gesture detection display
- **FPS monitoring** for performance tracking
- **Easy to use** - just run the script and show your hand to the camera
- **Cross-platform** compatibility (Windows, macOS, Linux)

## ⚙️ How It Works

### System Initialization

1. **MediaPipe Hands Setup**: Initializes the MediaPipe solution for hand landmark detection with:
   - Single hand tracking (`max_num_hands=1`)
   - Confidence thresholds for detection (0.7) and tracking (0.5)
   - Real-time video mode (`static_image_mode=False`)

2. **Webcam Initialization**: Opens the default webcam (`cv2.VideoCapture(0)`) for video input

3. **Screen Dimensions**: Retrieves screen width and height using `pyautogui.size()` for boundary calculations

4. **Gesture Parameters**: Sets up scrolling speed, cooldown timers, and movement thresholds

### Real-time Processing Loop

The system operates in a continuous loop that processes each webcam frame:

1. **Frame Capture**: Reads a frame from the webcam
2. **Mirror Effect**: Flips the frame horizontally for intuitive interaction (left/right movement matches user expectation)
3. **Color Conversion**: Converts BGR (OpenCV format) to RGB (MediaPipe required format)
4. **Hand Detection**: Processes the frame with MediaPipe to detect hand landmarks
5. **Gesture Recognition**:
   - When a hand is detected, extracts the index finger tip position (landmark 8)
   - Tracks vertical movement (Y-coordinate changes) between frames
   - Applies a threshold (`gesture_threshold = 20 pixels`) to distinguish intentional gestures from noise
   - Implements a cooldown period (`scroll_cooldown = 0.1 seconds`) to prevent excessive scrolling
6. **Action Execution**:
   - **Downward movement** (dy > threshold) → Scroll up (`pyautogui.scroll(positive_value)`)
   - **Upward movement** (dy < -threshold) → Scroll down (`pyautogui.scroll(negative_value)`)
7. **Visual Feedback**:
   - Draws detected hand landmarks on the video feed
   - Displays current gesture status (SCROLL_UP, SCROLL_DOWN, NO_GESTURE)
   - Shows real-time FPS (frames per second) for performance monitoring
   - Displays user instructions on screen
8. **Display & Control**: Shows the processed frame in a window and checks for 'q' key press to exit
9. **Cleanup**: Releases webcam resources and closes windows when terminated

### Gesture Logic Details

The scroll detection algorithm works by:

- **Tracking Movement**: Comparing current index finger Y-position with previous frame's position
- **Threshold Filtering**: Only registering movements exceeding `gesture_threshold` pixels
- **Direction Mapping**: 
  - Positive Y movement (finger moving down) = Scroll up (negative pyautogui.scroll value)
  - Negative Y movement (finger moving up) = Scroll down (positive pyautogui.scroll value)
- **Cooldown Mechanism**: Prevents gesture spamming by requiring a minimum time between recognized gestures

## 🛠️ Installation

Follow these steps to set up the hand detection system on your local machine:

### Prerequisites
Before you begin, ensure you have:
- **Python 3.7 or higher** installed on your system
- A **webcam** (built-in laptop camera or external USB webcam)
- **pip** package manager (comes with Python installation)

### Step-by-Step Installation

1. **Clone the Repository**
   Open your terminal or command prompt and run:
   ```bash
   git clone https://github.com/KarthikDulla/hand_detection_for_ppt.git
   cd hand_detection_for_ppt
   ```

2. **Install Required Dependencies**
   Install all necessary Python packages:
   ```bash
   pip install opencv-python mediapipe pyautogui numpy
   ```

   > 💡 **Alternative**: If you prefer using a requirements file, create one with the above packages and run:
   > ```bash
   > pip install -r requirements.txt
   > ```

3. **Verify Installation**
   Test that all packages are installed correctly by running:
   ```bash
   python -c "import cv2, mediapipe, pyautogui, numpy; print('All packages imported successfully!')"
   ```

### Troubleshooting Tips
- **Camera not detected**: Ensure your webcam is properly connected and not being used by another application
- **Permission issues**: On macOS, you may need to grant camera access to your terminal/Python IDE
- **Package installation fails**: Try upgrading pip first: `pip install --upgrade pip`

## ▶️ Usage

Follow these steps to use the hand gesture controller:

### Step-by-Step Usage Guide

1. **Prepare Your Environment**
   - Ensure your webcam is connected and functioning properly
   - Close any applications that might be using the webcam to avoid conflicts
   - Ensure good lighting in your environment for optimal hand detection

2. **Launch the Application**
   Navigate to the project directory and run:
   ```bash
   python gesture_scroller.py
   ```

3. **Position Yourself**
   - Sit comfortably at your desk with your webcam facing you
   - Ensure your hands are visible within the camera frame
   - The application will automatically start processing video feed

4. **Use Hand Gestures**
   Once the application is running:
   - **Show your hand** to the webcam (palm facing camera works best)
   - **Move your index finger vertically** to control scrolling:
     - **Move finger DOWN** → Scrolls UP in the active window
     - **Move finger UP** → Scrolls DOWN in the active window
   - The gesture detection is most sensitive with smooth, deliberate movements

5. **Monitor Feedback**
   The application window will show:
   - Live video feed with hand landmarks overlay
   - Current gesture status (SCROLL_UP, SCROLL_DOWN, or NO_GESTURE)
   - Real-time FPS (frames per second) indicator
   - On-screen instructions

6. **Exit the Application**
   - Press the **'q' key** on your keyboard to quit the application
   - Alternatively, close the application window
   - The system will automatically release webcam resources

### 💡 Usage Tips & Best Practices

- **For PowerPoint Presentations**: 
  - Make sure your PowerPoint window is active and in focus before using gestures
  - Practice the gestures beforehand to get comfortable with the sensitivity
  
- **Optimal Gesture Technique**:
  - Move your finger smoothly and deliberately, not too fast or too slow
  - Keep your hand relatively still except for the index finger movement
  - Start with your finger near the center of the frame for best results

- **Troubleshooting**:
  - **No response**: Check if the webcam LED is on (indicating it's active)
  - **Jittery scrolling**: Try slower, more deliberate finger movements
  - **False triggers**: Ensure no other objects are moving in the background
  - **Poor detection**: Improve lighting or adjust your distance from the camera

- **Performance Notes**:
  - The application runs best when not competing for CPU with other intensive applications
  - Closing unnecessary browser tabs or applications can improve responsiveness

## 🔧 Technical Details

### Libraries Used
- **OpenCV (cv2)**: Webcam access, image processing, and display
- **MediaPipe**: Machine learning pipeline for hand landmark detection
- **PyAutoGUI**: Cross-platform GUI automation for scrolling/keyboard control
- **NumPy**: Numerical operations for array handling
- **Time**: FPS calculation and gesture cooldown timing
- **Random**: Used in the included number guessing game (separate feature)

### Hand Landmark Points
The system specifically uses:
- **Landmark 8**: Index finger tip (primary tracking point for gestures)
- MediaPipe detects 21 hand landmarks total, but only the index finger tip is utilized for scroll detection

### Performance Considerations
- **Frame Processing**: Runs at webcam FPS (typically 20-30 FPS)
- **Latency**: Minimal delay between gesture and action (~100-200ms including processing)
- **Resource Usage**: Moderate CPU usage due to MediaPipe inference
- **Accuracy**: Reliable for intentional gestures; resistant to minor hand shaking

### Current Limitations
- Single hand tracking only (multi-hand support planned)
- Vertical scroll gestures only (horizontal/planned gestures in development)
- Requires consistent lighting for optimal detection
- Best performance with hand relatively close to camera

## 🚀 Future Enhancements

Planned features for future versions:
- [ ] **Slide Navigation Gestures**: 
  - Swipe left/right for previous/next slide
  - Pinch to zoom in/out
  - Fist click for slide advancement
- [ ] **Pointer Control**: 
  - Index finger positioning to move cursor
  - Gesture-based click/drag functionality
- [ ] **Gesture Customization**:
  - Configurable gesture mappings
  - Adjustable sensitivity settings
- [ ] **Multi-hand Support**:
  - Different gestures for each hand
  - Two-handed combination gestures
- [ ] **UI Improvements**:
  - Settings panel for real-time adjustment
  - Gesture tutorial mode
  - Dark/light theme options
- [ ] **Compatibility**:
  - Specific PowerPoint optimization
  - Support for Google Slides, Keynote
  - Integration with presentation remotes protocols

## 🎯 Use Cases

- **Presentations**: Hands-free slide navigation during lectures, meetings, or conferences
- **Accessibility**: Alternative control method for users with limited mobility
- **Demos**: Clean, professional presentation flow without visible controllers
- **Education**: Interactive teaching tools where instructors need to move freely
- **Entertainment**: Gesture-controlled media playback or gallery navigation

## 📝 Notes

- Ensure good lighting for optimal hand detection
- Keep hand within webcam's field of view
- The mirror effect makes interactions intuitive (your left movement appears as left on screen)
- Gesture cooldown prevents accidental multiple triggers
- Press 'q' anytime to exit the application safely

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some amazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Google MediaPipe](https://mediapipe.dev/) for the hand tracking solution
- [OpenCV](https://opencv.org/) for computer vision capabilities
- [PyAutoGUI](https://pyautogui.readthedocs.io/) for GUI automation
- Inspired by various hand gesture control projects in the open-source community

---

*Developed with ❤️ for hands-free presentation control*

*Last updated: October 2026*