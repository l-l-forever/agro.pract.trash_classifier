# ==========================
# Нейронная сеть для распознавания мусора
# TensorFlow + Kaggle/локальный датасет
# ==========================

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# Папка с датасетом.
# Для локального запуска картинки нужно положить так:
# D:\labaivanchuk\datasets\garbage\train
# D:\labaivanchuk\datasets\garbage\val
#
# Для Kaggle Notebook этот путь можно заменить на путь к датасету Kaggle, например:
# /kaggle/input/garbage-classification/Garbage classification/Garbage classification
DATASET_PATH = Path("datasets/garbage")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10


def main() -> None:
    train_path = DATASET_PATH / "train"
    val_path = DATASET_PATH / "val"

    # Перед обучением проверяем, что датасет имеет нужную структуру.
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(
            "Dataset must contain train and val folders.\n"
            f"Expected train: {train_path}\n"
            f"Expected val:   {val_path}"
        )

    # Загружаем обучающие изображения. Названия папок автоматически становятся классами.
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_path,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    # Загружаем валидационные изображения. Перемешивание выключено для стабильной проверки.
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_path,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)

    print("Garbage classes:")
    print(class_names)

    # Prefetch ускоряет загрузку данных во время обучения модели.
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=autotune)
    val_ds = val_ds.prefetch(buffer_size=autotune)

    # Аугментация создаёт слегка изменённые копии изображений во время обучения.
    # Это помогает модели лучше работать с новыми фотографиями.
    data_augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ]
    )

    # MobileNetV2 — это уже предобученная нейронная сеть.
    # Она умеет выделять признаки изображений благодаря обучению на ImageNet.
    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
    )

    # Замораживаем предобученную основу, чтобы сначала обучались только наши новые слои.
    base_model.trainable = False

    # Собираем итоговую модель:
    # аугментация -> нормализация -> предобученная основа -> слои классификации.
    model = models.Sequential(
        [
            data_augmentation,
            layers.Rescaling(1.0 / 255),
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.3),
            layers.Dense(128, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    # Компилируем модель: выбираем оптимизатор, функцию ошибки и метрику качества.
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    # Ранняя остановка прекращает обучение, если качество на валидации перестало улучшаться.
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
    )

    # Сохраняем лучшую модель по точности на валидационной выборке.
    checkpoint = ModelCheckpoint(
        "trash_classifier_best.keras",
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
    )

    # Запускаем обучение нейронной сети.
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[early_stop, checkpoint],
    )

    # Оцениваем обученную модель на валидационных данных.
    loss, acc = model.evaluate(val_ds)
    print(f"\nValidation accuracy: {acc:.4f}")
    print(f"Validation loss: {loss:.4f}")

    # Строим и сохраняем графики обучения для отчёта.
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("Accuracy")
    plt.legend(["Train", "Validation"])

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("Loss")
    plt.legend(["Train", "Validation"])

    plt.tight_layout()
    plt.savefig("training_history.png")
    plt.show()

    # Сохраняем финальную модель после обучения.
    model.save("trash_classifier_final.keras")
    print("Saved model: trash_classifier_final.keras")
    print("Saved best model: trash_classifier_best.keras")
    print("Saved training plot: training_history.png")


if __name__ == "__main__":
    main()
