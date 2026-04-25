from src.data.load_data import load_training_data
from src.models.model import build_cnn_model
from src.utils.visualization import plot_history

# Load data
X, y = load_training_data("data/raw/train.csv")

# Build model
model = build_cnn_model()

# Train
history = model.fit(
    X,
    y,
    batch_size=64,
    epochs=25,
    validation_split=0.2
)

# Save model
model.save("outputs/models/cnn_model.h5")

# Plot training curves
plot_history(history)
