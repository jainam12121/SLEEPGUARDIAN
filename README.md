"Drowsiness Detection System: Flask app detecting driver drowsiness using OpenCV and Dlib, alerting via SMS with Twilio and MongoDB for user data." Description:

This project is a comprehensive Flask web application designed to detect drowsiness in drivers and send alerts via SMS when drowsiness is detected. It leverages computer vision techniques to analyze video frames for signs of drowsiness, such as low eye aspect ratio (EAR) and mouth aspect ratio (MAR), indicative of drowsiness. The system is integrated with MongoDB for user data management and Twilio for sending SMS alerts.

Key Features:

Drowsiness Detection: Utilizes OpenCV and Dlib libraries to detect faces and analyze facial landmarks for EAR and MAR. Session Management: Implements Flask-Session for secure session management, supporting both filesystem and Redis storage. Database Integration: Connects to MongoDB Atlas for storing user and organization data, with support for indexing and unique constraints. User Authentication: Includes login functionality with email validation and password hashing, supporting both individual and organization users. SMS Alerts: Integrates with Twilio to send SMS alerts to users when drowsiness is detected, ensuring immediate action. Web Interface: Provides a user-friendly web interface for user registration, login, and viewing drowsiness detection results. Technologies Used:

Flask: Web framework for Python OpenCV: Computer vision library for image and video processing Dlib: Toolkit for machine learning and computer vision MongoDB: NoSQL database for storing user and organization data Twilio: Communication platform for sending SMS alerts Flask-Session: Session management for Flask Getting Started:

To run this project locally, clone the repository and install the required dependencies listed in the requirements.txt file. Ensure you have a MongoDB Atlas account and Twilio account for database and SMS services. Update the configuration files with your MongoDB and Twilio credentials.

Contributions:

Contributions, issues, and feature requests are welcome. Please feel free to open an issue or submit a pull request.
