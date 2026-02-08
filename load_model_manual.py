from tensorflow.keras.models import load_model

# Load model lama .h5 (TF Keras 2.13)
model = load_model("model_motility.h5", compile=False)
print("✅ Model berhasil di-load di TF Keras 2.13!")

# Save ulang ke format Keras 3 / SavedModel
model.save("model_motility.keras")  # Keras 3 compatible
print("✅ Model berhasil disimpan ke format Keras 3!")
