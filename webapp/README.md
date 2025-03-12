# MI Prediction Model

This project is a Flask web application that uses a machine learning model to predict the risk of Myocardial Infarction (MI). The model is loaded from a `.pkl` file, and patient data is collected via a user-friendly form.

## Prerequisites

- Python 3.x
- [Flask](https://flask.palletsprojects.com/)
- [joblib](https://joblib.readthedocs.io/)
- numpy

## Setup

1. **Clone this repository** and navigate to the project folder.

2. **Create and activate a virtual environment:**

   ```bash
   python3 -m venv venv
   source venv/bin/activate   # on Linux/macOS
   # or
   venv\Scripts\activate      # on Windows

3. **Install dependencies**
pip install flask joblib numpy

4. **Model File Setup**
Place your trained model file (mi_best_model.pkl) in the project directory.
Note: In app.py, update the MODEL_PATH variable with your specific file path if it's not in the root directory. For example:

MODEL_PATH = "/path/to/your/mi_best_model.pkl"

5. **Running the Application**
Start the Flask application with: python app.py

http://127.0.0.1:5000




