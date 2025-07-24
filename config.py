import os

IMAGE_DIR = r"C:\Users\MAXIMUS8\Downloads\archive\fashion-dataset\images"
OUTPUT_DIR = r"C:\Users\MAXIMUS8\Downloads\archive\fashion-dataset\output"
DESCRIPTION_CSV = r"C:\Users\MAXIMUS8\Downloads\archive\fashion-dataset\fashion_text_descriptions.csv"
BATCH_SIZE = 16  # You can try a higher batch size now
MODEL_NAME = "google/siglip2-base-patch16-224"
MAX_TEXT_LENGTH = 64
VALIDATION_SPLIT = 0.1
RANDOM_SEED = 42
EPOCHS = 3
LEARNING_RATE = 2e-5