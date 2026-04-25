# Digit Recognizer
Refactored code and visualizations from my Jupyter notebook submission to Kaggle's recurring Digit Recognizer machine learning competition.

## Objective

This project is based on the Digit Recognizer competition on Kaggle.  The task is to correctly predict the correct label (0–9) for hand-drawn digits from 28x28 grayscale images' pixel values (MNIST dataset format).



* [Approach](#Approach)
* [Results](#Results)
* [Project Structure](#project-structure)
* [Setup](#setup)
* [Data](#data)
* [About](#about)
* [Train Models](#train-models)
* [Key Insights](#key-insights)
* [Future Improvements](#future-improvements)
* [Tech Stack](#tech-stack)

---

## Approach

### 1. Exploratory Data Analysis (EDA)

* Visualized sample digits
* Checked class balance (uniform distribution)
* Verified pixel intensity ranges

### 2. Preprocessing

* Normalized pixel values (0–255 → 0–1)
* Reshaped data for CNN input (28×28×1)
* Train/validation split

### 3. Models

#### Advanced Model (Convolutional Neural Network)

* 2 Convolutional layers + MaxPooling
* Dropout for regularization
* Fully connected dense layers

**Final CNN Performance:**

* Kaggle Score: ~0.97839

---

## Results

| Model               | Accuracy |
| ------------------- | -------- |
| CNN                 | 0.98     |

---

## Project Structure

* `notebooks/` → experimentation and EDA
* `src/` → reusable pipeline code
* `outputs/` → trained models + predictions

---

## Setup

```bash id="setup-dig"
git clone https://github.com/yourusername/digit-recognizer.git
cd digit-recognizer
pip install -r requirements.txt
```

---

## Data

Download the dataset from Kaggle and place files in:

```id="data-path"
data/raw/
```

(Data is not included due to Kaggle competition rules.)

---

## Train Models

### CNN model

```bash id="cnn-run"
python src/models/train_cnn.py
```

---

## Key Insights

* Even simple models perform well due to clean dataset
* CNN significantly outperforms classical ML

---

## Future Improvements

* Hyperparameter tuning (learning rate, filters)
* New layers
* Data augmentation
* Ensemble multiple CNNs
* Deploy model as an API or web app

---

## Tech Stack

* Python
* NumPy, Pandas
* TensorFlow
* Keras

---

## Acknowledgements

Competition hosted on Kaggle (MNIST dataset)
