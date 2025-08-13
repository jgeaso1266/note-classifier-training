import argparse
import json
import os
import typing as ty
import tensorflow as tf
from tensorflow import keras
from keras import layers, Model
import numpy as np
from tqdm import tqdm
import cv2

# Model type constants (kept for compatibility, though this is a different task)
single_label = "MODEL_TYPE_SINGLE_LABEL_CLASSIFICATION"
multi_label = "MODEL_TYPE_MULTI_LABEL_CLASSIFICATION"
labels_filename = "labels.txt"

TFLITE_OPS = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
    tf.lite.OpsSet.SELECT_TF_OPS,  # enable TensorFlow ops.
]

# For piano detection, we have 88 keys
NUM_KEYS = 88
DEFAULT_EPOCHS = 200
IMG_SIZE = (480, 640)
NUM_FRAMES = 5  # Number of consecutive frames to use as input

def parse_args():
    """Returns dataset file, model output directory, and num_epochs if present."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_file", dest="data_json",
                        type=str, required=True)
    parser.add_argument("--model_output_directory", dest="model_dir",
                        type=str, required=True)
    parser.add_argument("--num_epochs", dest="num_epochs", type=int)
    args = parser.parse_args()
    
    return args.data_json, args.model_dir, args.num_epochs


def parse_piano_dataset_from_json(
    filename: str
) -> ty.Tuple[ty.List[str], ty.List[np.ndarray]]:
    """Load and parse JSON file to return image filenames and corresponding
    piano key labels (88 binary values).
    
    Args:
        filename: JSONLines file containing filenames and piano key annotations
    """
    image_filenames = []
    key_labels = []
    
    with open(filename, "rb") as f:
        for line in f:
            json_line = json.loads(line)
            image_filenames.append(json_line["image_path"])
            
            # Adjust this based on your actual data format
            labels = [0] * NUM_KEYS
            for annotation in json_line["classification_annotations"]:
                labels[int(annotation["annotation_label"]) - 21] = 1
            
            key_labels.append(np.array(labels, dtype=np.float32))
    return image_filenames, key_labels

class PianoModelBlock(layers.Layer):
    """Residual block for the piano model."""
    
    def __init__(self, out_dim, kernel_size=(3, 3), strides=(1, 1), dropout_rate=0.0):
        super().__init__()
        self.out_dim = out_dim
        self.kernel_size = kernel_size
        self.strides = strides
        self.dropout_rate = dropout_rate
        
    def build(self, input_shape):
        in_dim = input_shape[-1]
        padding = "same"
        
        # Main path
        self.conv1 = layers.Conv2D(self.out_dim, 1, strides=1, use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.act1 = layers.LeakyReLU()
        
        self.conv2 = layers.Conv2D(self.out_dim, self.kernel_size, strides=self.strides, 
                                   padding=padding, use_bias=False)
        self.bn2 = layers.BatchNormalization()
        self.act2 = layers.LeakyReLU()
        self.dropout = layers.Dropout(self.dropout_rate)
        
        self.conv3 = layers.Conv2D(self.out_dim, 1, use_bias=False)
        self.bn3 = layers.BatchNormalization()
        
        # Shortcut path
        self.downsample = keras.Sequential([
            layers.Conv2D(self.out_dim, self.kernel_size, strides=self.strides,
                         padding=padding, use_bias=False),
            layers.BatchNormalization()
        ])
        
        self.final_act = layers.LeakyReLU()
        
    def call(self, x, training=False):
        # Main path
        out = self.conv1(x)
        out = self.bn1(out, training=training)
        out = self.act1(out)
        
        out = self.conv2(out)
        out = self.bn2(out, training=training)
        out = self.act2(out)
        out = self.dropout(out, training=training)
        
        out = self.conv3(out)
        out = self.bn3(out, training=training)
        
        # Shortcut
        shortcut = self.downsample(x, training=training)
        
        # Residual connection
        out = self.final_act(out + shortcut)
        return out

def build_piano_model(
    input_shape: ty.Tuple[int, int, int, int] = (NUM_FRAMES, 480, 640, 1)
) -> Model:
    """Build the piano detection model.
    
    Args:
        input_shape: Shape of input (num_frames, height, width, channels)
    """
    inputs = layers.Input(shape=input_shape)
    
    # Aggregate input frames using 3D convolutions
    x = layers.Conv3D(8, kernel_size=(1, 3, 3), padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    
    x = layers.Conv3D(16, kernel_size=(3, 3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    
    x = tf.pad(x, paddings=[[0, 0], [0, 0], [1, 1], [1, 1], [0, 0]])
    x = layers.Conv3D(32, kernel_size=(NUM_FRAMES, 3, 3), padding="valid")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    print(x.shape)
    # Squeeze the time dimension
    x = tf.squeeze(x, axis=1)
    
    # Process the last frame separately
    last_frame = inputs[:, -1, :, :, :]
    pred_frame = layers.Conv2D(32, (3, 3), padding="same")(last_frame)
    pred_frame = layers.BatchNormalization()(pred_frame)
    pred_frame = layers.LeakyReLU()(pred_frame)
    
    # Combine aggregated frames with processed last frame
    x = layers.Add()([x, pred_frame])
    
    # Apply residual blocks with progressive downsampling
    x = PianoModelBlock(32, kernel_size=(3, 3), strides=(2, 2), dropout_rate=0.2)(x)
    x = PianoModelBlock(64, kernel_size=(3, 3), strides=(2, 2), dropout_rate=0.2)(x)
    x = PianoModelBlock(128, kernel_size=(3, 3), strides=(2, 1), dropout_rate=0.2)(x)
    x = PianoModelBlock(128, kernel_size=(3, 3), strides=(2, 1), dropout_rate=0.2)(x)
    x = PianoModelBlock(256, kernel_size=(3, 3), strides=(2, 1), dropout_rate=0.0)(x)
    
    # Final processing
    x = layers.Permute((2, 1, 3))(x)  # Swap height and width dimensions
    x = layers.Conv2D(1, 1)(x)
    x = tf.squeeze(x, axis=-1)
    
    x = layers.Conv1D(256, 3, padding="same")(x)
    x = layers.Flatten()(x)
    
    # Output layer for 88 piano keys
    outputs = layers.Dense(NUM_KEYS, activation="sigmoid")(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model


def load_and_preprocess_data(
    filenames: ty.List[str],
    labels: ty.List[np.ndarray],
    num_frames: int = NUM_FRAMES,
    img_size: ty.Tuple[int, int] = IMG_SIZE,
    batch_size: int = 16,
    train_split: float = 0.9
) -> ty.Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Load images and create TensorFlow datasets for training and validation.
    
    Args:
        filenames: List of image file paths
        labels: List of label arrays (88 binary values each)
        num_frames: Number of consecutive frames to use
        img_size: Target image size
        batch_size: Batch size for training
        train_split: Proportion of data to use for training
    """
    # Filter out samples that don't have enough preceding frames
    valid_indices = list(range(num_frames - 1, len(filenames)))
    
    def load_frames(idx):
        """Load a sequence of frames."""
        frames = []
        for i in range(idx - num_frames + 1, idx + 1):
            img = cv2.imread(filenames[i])
            if img is None:
                img = np.zeros(img_size, dtype=np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (img_size[1], img_size[0]))
            img = img.astype(np.float32) / 255.0
            frames.append(img[..., np.newaxis])
        return np.stack(frames, axis=0)
    
    # Load all data into memory (for small datasets)
    # For large datasets, consider using tf.data.Dataset.from_generator
    X = []
    y = []
    
    print(f"Loading {len(valid_indices)} samples...")
    for idx in tqdm(valid_indices):
        frames = load_frames(idx)
        X.append(frames)
        y.append(labels[idx])
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"Data shapes - X: {X.shape}, y: {y.shape}")
    print(f"Expected shapes - X: (N, {num_frames}, {img_size[0]}, {img_size[1]}, 1), y: (N, {NUM_KEYS})")
    
    # Ensure labels have correct shape
    if len(y.shape) == 1:
        print(f"Warning: Reshaping labels from {y.shape} to ({len(y)}, {NUM_KEYS})")
        y = y.reshape(-1, NUM_KEYS)
    elif y.shape[1] != NUM_KEYS:
        print(f"Error: Expected {NUM_KEYS} keys, got {y.shape[1]}")
        raise ValueError(f"Label shape mismatch: expected {NUM_KEYS} keys, got {y.shape[1]}")
    
    # Split into train and validation
    num_train = int(len(X) * train_split)
    indices = np.random.permutation(len(X))
    
    X_train = X[indices[:num_train]]
    y_train = y[indices[:num_train]]
    X_val = X[indices[num_train:]]
    y_val = y[indices[num_train:]]
    
    print(f"Train set: {X_train.shape}, Val set: {X_val.shape}")
    
    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    
    # Data augmentation for training
    def augment(x, y):
        # Random brightness adjustment
        x = tf.image.random_brightness(x, 0.2)
        # Random contrast adjustment
        x = tf.image.random_contrast(x, 0.7, 1.3)
        # Add noise
        noise = tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=0.05)
        x = x + noise
        x = tf.clip_by_value(x, 0.0, 1.0)
        return x, y
    
    train_dataset = train_dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.shuffle(buffer_size=1000)
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    
    val_dataset = val_dataset.batch(batch_size)
    val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
    
    return train_dataset, val_dataset


def weighted_binary_crossentropy(y_true, y_pred):
    """Custom weighted binary crossentropy loss.
    Gives more weight to positive samples (pressed keys).
    """
    # Weight for positive samples (pressed keys)
    pos_weight = 9.0
    # Weight for negative samples (not pressed keys)
    neg_weight = 1.0
    
    loss = -(pos_weight * y_true * tf.math.log(y_pred + 1e-7) + 
             neg_weight * (1 - y_true) * tf.math.log(1 - y_pred + 1e-7))
    return tf.reduce_mean(loss)


def save_labels(labels: ty.List[str], model_dir: str) -> None:
    """Save a labels.txt file with piano key indices (MIDI note numbers 21-108)."""
    filename = os.path.join(model_dir, labels_filename)
    with open(filename, "w") as f:
        for label in labels[:-1]:
            f.write(label + "\n")
        f.write(labels[-1])

def save_model(
    model: Model,
    model_dir: str,
    model_name: str,
    input_shape: ty.Tuple[int, int, int, int],
    target_shape: ty.Tuple[int]
) -> None:
    """Save model as a TFLite model.
    
    Args:
        model: Trained model
        model_dir: Output directory for model artifacts
        model_name: Name of saved model
        input_shape: Input shape for the model
    """
    # Create a representative dataset for quantization (optional)
    def representative_dataset():
        for _ in range(100):
            data = np.random.rand(1, *input_shape).astype(np.float32)
            yield [data]
    
    # Convert to TFLite
    input = tf.keras.Input(input_shape, batch_size=1, dtype=tf.uint8)
    output = model(input, training=False)
    wrapped_model = tf.keras.Model(inputs=input, outputs=output)
    converter = tf.lite.TFLiteConverter.from_keras_model(wrapped_model)
    converter.target_spec.supported_ops = TFLITE_OPS
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # Uncomment for full integer quantization if needed
    # converter.representative_dataset = representative_dataset
    
    tflite_model = converter.convert()
    
    # Save the model
    filename = os.path.join(model_dir, f"{model_name}.tflite")
    with open(filename, "wb") as f:
        f.write(tflite_model)
    
    print(f"Model saved to {filename}")


if __name__ == "__main__":
    DATA_JSON, MODEL_DIR, NUM_EPOCHS = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Validate epochs
    epochs = DEFAULT_EPOCHS if NUM_EPOCHS is None or NUM_EPOCHS <= 0 else int(NUM_EPOCHS)
    
    # Read dataset file
    print("Loading dataset...")
    image_filenames, key_labels = parse_piano_dataset_from_json(DATA_JSON)
    
    # Create datasets
    print("Creating training and validation datasets...")
    train_dataset, val_dataset = load_and_preprocess_data(
        image_filenames,
        key_labels,
        num_frames=NUM_FRAMES,
        img_size=IMG_SIZE,
        batch_size=16,
        train_split=0.9
    )
    
    # Build and compile model
    print("Building model...")
    input_shape = (NUM_FRAMES, IMG_SIZE[0], IMG_SIZE[1], 1)
    target_shape = (NUM_KEYS)
    model = build_piano_model(input_shape)
    
    # Compile with custom loss and metrics
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=weighted_binary_crossentropy,
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    
    # Print model summary
    model.summary()
    
    # Setup callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6
        ),
    ]
    
    # Train the model
    print(f"Training for {epochs} epochs...")
    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=val_dataset,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save labels.txt file
    print("Saving labels...")
    labels = [f"{i+21}" for i in range(NUM_KEYS)]
    save_labels(labels, MODEL_DIR)
    
    # Convert and save as TFLite
    print("Converting to TFLite...")
    save_model(
        model,
        MODEL_DIR,
        "piano_detection_model",
        input_shape,
        target_shape,
    )
    
    print(f"Training complete! Model saved to {MODEL_DIR}")
    
    # Print final metrics
    print("\nFinal Training Metrics:")
    for metric_name in history.history.keys():
        if not metric_name.startswith('val_'):
            final_value = history.history[metric_name][-1]
            print(f"  {metric_name}: {final_value:.4f}")
    
    print("\nFinal Validation Metrics:")
    for metric_name in history.history.keys():
        if metric_name.startswith('val_'):
            final_value = history.history[metric_name][-1]
            print(f"  {metric_name}: {final_value:.4f}")
