Drowsiness Detection
This repository hosts a robust and real-time drowsiness detection system designed to enhance road safety by identifying signs of driver fatigue. Utilizing advanced computer vision techniques and machine learning models, the system continuously monitors facial features to detect key indicators of drowsiness, such as eye closure, blink frequency, and head pose. When signs of drowsiness are detected, the system triggers alerts to help prevent accidents.

Features
Real-time Face Detection: Employs a pre-trained Dlib frontal face detector to accurately identify and track faces in real-time video streams.

Facial Landmark Detection: Utilizes a 68-point facial landmark model to precisely locate and monitor critical regions of the face, including the eyes and mouth.

Eye Aspect Ratio (EAR) Calculation: Measures the ratio of eye dimensions to detect prolonged eye closure, a primary indicator of drowsiness.

Blink Detection: Counts blinks by analyzing changes in the Eye Aspect Ratio (EAR) over consecutive frames.

PERCLOS Calculation: Calculates the Percentage of Eye Closure (PERCLOS) over time to assess the level of drowsiness.

Head Pose Estimation: Detects unusual head tilts using the PnP (Perspective-n-Point) algorithm, signaling potential fatigue-related head movements.

Customizable Thresholds: Adjustable thresholds for EAR, PERCLOS, and head pose to fine-tune detection sensitivity based on real-world testing.

Visual Feedback and Alerts: Provides real-time visual indicators, including facial landmarks and alert messages, within the video stream. An optional sound alert can be integrated for immediate attention.

Installation
Clone this repository:

sh
Copy code
git clone https://github.com/your-username/Drowsiness-Detection.git
cd Drowsiness-Detection
Install the required dependencies:

sh
Copy code
pip install -r requirements.txt
Download the necessary Dlib model files (e.g., shape_predictor_68_face_landmarks.dat) and place them in the appropriate directory.

Run the system:

sh
Copy code
python drowsiness_detection.py
Usage
Simply run the script and the system will start capturing video from your webcam. It will continuously monitor facial landmarks for signs of drowsiness, displaying real-time feedback and issuing alerts if drowsiness is detected.

Customization
The system is highly customizable. You can adjust thresholds, change the detection models, and integrate additional features like sound alerts or logging for long-term monitoring.

Contributing
Contributions are welcome! Please feel free to submit issues or pull requests to enhance the functionality, performance, or usability of this project.

License
This project is licensed under the MIT License - see the LICENSE file for details.

By implementing this Drowsiness Detection system, we can contribute to reducing the risk of accidents caused by driver fatigue, making roads safer for everyone.

Feel free to adjust any specific details, especially the installation instructions, based on your project's configuration!
