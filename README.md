# Facial Emotion Detection using CNN

A deep learning project that detects and classifies human emotions from facial expressions in real-time using Convolutional Neural Networks (CNNs). Built using TensorFlow, Keras, and OpenCV, the model is trained on the FER-2013 dataset and supports real-time webcam-based emotion recognition.

## Features

- Custom CNN architecture for multi-class emotion detection  
- Trained on the FER-2013 dataset  
- Real-time detection via webcam using OpenCV  
- Seven emotion classes: Angry, Disgusted, Fearful, Happy, Neutral, Sad, Surprised  
- Modular, easy-to-understand code for future enhancement and experimentation

## Tech Stack

- Python 3.x  
- TensorFlow / Keras  
- OpenCV  
- NumPy  
- Matplotlib  

## Applications

- Mental health & therapy tools  
- Smart surveillance systems  
- E-learning sentiment tracking  
- Human-computer interaction  
- UX analysis in digital products

## Requirements

- tensorflow==2.13.0
- opencv-python
- numpy
- pandas

## How to Run

1. **Clone the repository**
```bash
git clone https://github.com/your-username/FacialEmotionRecog.git
cd FacialEmotionRecog
```

2. **Create and activate a virtual environment**
```bash
python3 -m venv facial_env310
source facial_env310/bin/activate
```

3. **Install the required libraries**
If you have a `requirements.txt` file:
```bash
pip install -r requirements.txt
```

If not, install manually:
```bash
pip install tensorflow opencv-python pandas numpy matplotlib
```

4. **Train the model**
> Skip this step if `emotion_detection_model_gsn.h5` is already available.
```bash
python train_model.py
```

5. **Run the emotion detection system**
```bash
python main.py
```

## Dataset

- **FER-2013**: Publicly available dataset containing facial emotion expressions.  
  Source: [Kaggle - FER-2013](https://www.kaggle.com/datasets/msambare/fer2013)

## Dependencies

- Python 3.10+
- TensorFlow
- OpenCV
- NumPy
- Pandas
- Matplotlib

## Author

**Unaiza Nouman** — https://www.linkedin.com/in/unaiza-nouman-2j9u0n6e/

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
