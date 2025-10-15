import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.optimizers import Adam

# === Configuration ===
MODEL_PATH = "hardware_model.h5"
DATASET_PATH = "hardware_dataset"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 15            # train longer for better accuracy
LEARNING_RATE = 1e-4   # slightly higher for fine-tuning stability

print("🔹 Loading existing model...")
model = load_model(MODEL_PATH)

# === Data augmentation ===
print("🔹 Preparing dataset with augmentation...")
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=25,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

num_classes = train_gen.num_classes
print(f"Detected {num_classes} component classes.")

# === Adjust model if class count changed ===
if model.output_shape[-1] != num_classes:
    print("Updating model output layer for new class count...")
    x = model.layers[-2].output
    new_output = Dense(num_classes, activation='softmax', name='new_output')(x)
    model = Model(inputs=model.input, outputs=new_output)

# === Freeze base model layers ===
print("🔹 Freezing base layers...")
for layer in model.layers[:-2]:
    layer.trainable = False

# === Compile ===
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# === Train ===
print("🚀 Starting fine-tuning...")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS
)

# === Save updated model ===
model.save(MODEL_PATH)
print(f"✅ Model updated and saved as {MODEL_PATH}")
