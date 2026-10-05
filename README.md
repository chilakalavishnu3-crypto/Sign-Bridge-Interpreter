# SignBridge – Indian Sign Language Interpreter 

SignBridge is a web-based application designed to help people communicate using Indian Sign Language (ISL) at public service counters such as railway stations and hospitals.

The system uses a webcam to recognize hand signs and converts them into text. Staff can also send replies using text, quick-reply buttons, and a 3D signing hand.

## Features

- Real-time sign recognition using a webcam
- Indian Sign Language (ISL) sign recognition
- Hand landmark detection using MediaPipe
- Machine learning based sign classification
- Converts recognized signs into text
- Supports English, Telugu, Tamil, Hindi, Kannada and Malayalam
- Text-to-speech support
- Quick replies for common requests
- 3D hand-based signing for staff responses
- Sign guide as a fallback when camera recognition is unavailable
- Teach a Sign feature for adding custom signs
- Railway and hospital communication modes

## How It Works

```text
Webcam
   ↓
MediaPipe Hand Detection
   ↓
21 Hand Landmarks × 2 Hands
   ↓
Feature Normalization
   ↓
Machine Learning Model
   ↓
Sign Prediction
   ↓
Sentence Formation
   ↓
Text / Speech
