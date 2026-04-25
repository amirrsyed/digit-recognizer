import pandas as pd
from tensorflow import keras
from src.data.load_data import load_test_data

# Load model
model = keras.models.load_model("outputs/models/cnn_model.h5")

# Load test data
X_test = load_test_data("data/raw/test.csv")

# Predict
predictions = model.predict(X_test)
labels = predictions.argmax(axis=1)

# Save submission
submission = pd.DataFrame({
    "ImageId": range(1, len(labels) + 1),
    "Label": labels
})

submission.to_csv("outputs/predictions/submission.csv", index=False)
