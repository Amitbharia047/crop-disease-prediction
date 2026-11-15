"""
train_cnn.py — Training & Evaluation Pipeline for AgriVision CNN
─────────────────────────────────────────────────────────────────
Trains the AgriVision Keras Convolutional Neural Network on the
PlantVillage Benchmark Crop Disease Dataset (38 classes).

Usage:
  # Train on full dataset
  python ml/train_cnn.py --dataset_dir /path/to/plantvillage --epochs 25

  # Quick pipeline validation & artifact compilation test (demo mode)
  python ml/train_cnn.py --demo
─────────────────────────────────────────────────────────────────
"""

import os
import sys
import json
import argparse
from datetime import datetime

# Add root directory to sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from ml.dataset import PLANT_VILLAGE_CLASSES, get_data_augmentation_pipeline, load_plant_village_dataset
from ml.cnn_model import build_crop_disease_cnn, build_transfer_learning_cnn, get_model_summary_dict


def parse_args():
    parser = argparse.ArgumentParser(description="Train AgriVision CNN on PlantVillage Dataset")
    parser.add_argument("--dataset_dir", type=str, default="./data/plantvillage",
                        help="Path to PlantVillage dataset directory")
    parser.add_argument("--epochs", type=int, default=25,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Initial learning rate")
    parser.add_argument("--transfer_learning", action="store_true",
                        help="Use MobileNetV2 backbone instead of custom ConvNet")
    parser.add_argument("--save_path", type=str, default="./ml/crop_disease_cnn.keras",
                        help="Path to save trained Keras model")
    parser.add_argument("--demo", action="store_true",
                        help="Run verification test with synthetic tensors (no external download needed)")
    return parser.parse_args()


def run_demo_training(save_path: str):
    """
    Simulates a rapid 2-epoch training cycle using synthetic leaf tensors to verify
    end-to-end model construction, tensor flow, optimizer backpropagation, and weight saving.
    """
    print("\n" + "=" * 65)
    print(" AGRIVISION DEEP LEARNING: CNN PIPELINE VERIFICATION MODE")
    print("=" * 65)

    try:
        import numpy as np
        import tensorflow as tf
    except ImportError:
        print("\n[NOTE] TensorFlow is not currently installed in this environment.")
        print("Saving pre-compiled model architecture specification and class index map...")
        export_class_indices(os.path.join(os.path.dirname(save_path), "class_indices.json"))
        specs = get_model_summary_dict()
        with open(os.path.join(os.path.dirname(save_path), "model_specs.json"), "w") as f:
            json.dump(specs, f, indent=2)
        print("✓ Architecture specifications exported to ml/model_specs.json")
        print("✓ Class index mapping exported to ml/class_indices.json")
        print("\nTo train with full GPU acceleration:")
        print("  pip install tensorflow matplotlib")
        print("  python ml/train_cnn.py --dataset_dir <path_to_plantvillage>")
        return

    num_classes = len(PLANT_VILLAGE_CLASSES)
    print(f"• Target Classes: {num_classes} PlantVillage crop disease categories")
    print("• Building AgriVision Deep CNN (Keras Sequential ConvNet)...")
    model = build_crop_disease_cnn(input_shape=(224, 224, 3), num_classes=num_classes)

    # Optimizer & Loss
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top_3_acc")]
    )
    model.summary()

    print("\n• Generating synthetic leaf batches (Batch size: 8, Resolution: 224x224x3)...")
    x_demo = np.random.randint(0, 256, size=(16, 224, 224, 3)).astype(np.float32)
    y_demo = tf.keras.utils.to_categorical(np.random.randint(0, num_classes, size=(16,)), num_classes)

    print("• Executing forward pass, loss computation, and backpropagation...")
    history = model.fit(x_demo, y_demo, epochs=2, batch_size=8, verbose=1)

    # Save model
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    model.save(save_path)
    print(f"✓ Model weights and computational graph saved to: {save_path}")

    # Save class indices
    indices_path = os.path.join(os.path.dirname(save_path), "class_indices.json")
    export_class_indices(indices_path)
    print(f"✓ Class index mapping saved to: {indices_path}")
    print("\n✓ Pipeline verification test passed successfully!")


def export_class_indices(dest_path: str):
    """Exports class index mapping to JSON for inference engines."""
    os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
    mapping = {idx: name for idx, name in enumerate(PLANT_VILLAGE_CLASSES)}
    with open(dest_path, "w") as f:
        json.dump(mapping, f, indent=2)


def main():
    args = parse_args()

    if args.demo or not os.path.exists(args.dataset_dir):
        if not args.demo:
            print(f"Dataset directory '{args.dataset_dir}' not found. Falling back to verification test.")
        run_demo_training(args.save_path)
        return

    try:
        import tensorflow as tf
        import matplotlib.pyplot as plt
    except ImportError:
        print("Error: tensorflow and matplotlib are required for training.")
        print("Run: pip install tensorflow matplotlib")
        sys.exit(1)

    print("\n" + "=" * 65)
    print(" AGRIVISION DEEP LEARNING: TRAINING ON PLANTVILLAGE DATASET")
    print("=" * 65)
    print(f"Dataset path: {args.dataset_dir}")
    print(f"Epochs:       {args.epochs}")
    print(f"Batch size:   {args.batch_size}")
    print(f"Initial LR:   {args.lr}")

    # 1. Load Dataset
    print("\n1. Loading dataset with Keras image preprocessing pipeline...")
    train_ds, val_ds = load_plant_village_dataset(
        args.dataset_dir,
        img_size=(224, 224),
        batch_size=args.batch_size,
        validation_split=0.2
    )

    # 2. Build Model
    num_classes = len(PLANT_VILLAGE_CLASSES)
    if args.transfer_learning:
        print("\n2. Building MobileNetV2 Transfer Learning Model...")
        model = build_transfer_learning_cnn(input_shape=(224, 224, 3), num_classes=num_classes)
    else:
        print("\n2. Building Custom Deep CNN (AgriVision-CNN v2)...")
        model = build_crop_disease_cnn(input_shape=(224, 224, 3), num_classes=num_classes)

    # 3. Compile
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top_3_acc")]
    )

    # 4. Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(args.save_path, monitor="val_accuracy", save_best_only=True, verbose=1)
    ]

    # 5. Train
    print("\n3. Starting model training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks
    )

    # 6. Export Class Indices
    indices_path = os.path.join(os.path.dirname(args.save_path), "class_indices.json")
    export_class_indices(indices_path)
    print(f"\n✓ Training completed. Best model saved to: {args.save_path}")
    print(f"✓ Class index mapping saved to: {indices_path}")


if __name__ == "__main__":
    main()

