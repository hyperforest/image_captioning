import os
import re

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization

from dataset import (decode_and_resize, load_captions_data, make_dataset, train_val_split)
from model import (ImageCaptioningModel, LRSchedule, TransformerDecoderBlock,
                       TransformerEncoderBlock, get_cnn_model)
from utils import generate_caption, load_model
from transformers import TFTransfoXLModel, TransfoXLConfig

seed = 111
np.random.seed(seed)
tf.random.set_seed(seed)

# Path to the images
IMAGES_PATH = "../dataset/Flickr8k_Dataset"
TEXT_PATH = "../dataset/Flickr8k_text/Flickr8k.token.txt"
WEIGHTS_PATH = '../weights_b0_transfo_epoch_001.h5'

# Desired image dimensions
IMAGE_SIZE = (32, 32)

# Vocabulary size
VOCAB_SIZE = 1000

# Fixed length allowed for any sequence
SEQ_LENGTH = 25

# Dimension for the image embeddings and token embeddings
EMBED_DIM = 32

# Per-layer units in the feed-forward network
FF_DIM = 32

# Other training parameters
BATCH_SIZE = 64
EPOCHS = 1
AUTOTUNE = tf.data.AUTOTUNE


# Load the dataset
captions_mapping, text_data = load_captions_data(TEXT_PATH, IMAGES_PATH, SEQ_LENGTH)

# Split the dataset into training and validation sets
train_data, valid_data = train_val_split(captions_mapping)
print("Number of training samples: ", len(train_data))
print("Number of validation samples: ", len(valid_data))

strip_chars = "!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"
strip_chars = strip_chars.replace("<", "")
strip_chars = strip_chars.replace(">", "")

def custom_standardization(input_string):
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(lowercase, "[%s]" % re.escape(strip_chars), "")

# Vectorizing the text data
vectorization = TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode="int",
    output_sequence_length=SEQ_LENGTH,
    standardize=custom_standardization,
)
vectorization.adapt(text_data)

# Data augmentation for image data
image_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomContrast(0.3),
    ]
)

def process_input(img_path, captions):
    return decode_and_resize(img_path, IMAGE_SIZE), vectorization(captions)

# Pass the list of images and the list of corresponding captions
train_dataset = make_dataset(
    list(train_data.keys()),
    list(train_data.values()),
    BATCH_SIZE, process_input
)

valid_dataset = make_dataset(
    list(valid_data.keys()),
    list(valid_data.values()),
    BATCH_SIZE, process_input
)

cnn_model = get_cnn_model(IMAGE_SIZE)
# encoder = TransformerEncoderBlock(embed_dim=EMBED_DIM, dense_dim=FF_DIM, num_heads=1)
encoder_config = TransfoXLConfig(
    vocab_size=VOCAB_SIZE, d_model=FF_DIM, d_embed=EMBED_DIM, d_inner=FF_DIM,
    n_head=1, d_head=16, n_layer=2
)
encoder = TFTransfoXLModel(encoder_config)
decoder = TransformerDecoderBlock(embed_dim=EMBED_DIM, ff_dim=FF_DIM, num_heads=2, \
    seq_length=SEQ_LENGTH, vocab_size=VOCAB_SIZE
)
caption_model = ImageCaptioningModel(
    cnn_model=cnn_model, encoder=encoder, decoder=decoder, image_aug=image_augmentation,
)

# Define the loss function
cross_entropy = keras.losses.SparseCategoricalCrossentropy(
    from_logits=False, reduction="none"
)

# EarlyStopping criteria
early_stopping = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)

# Create a learning rate schedule
num_train_steps = len(train_dataset) * EPOCHS
num_warmup_steps = num_train_steps // 15
lr_schedule = LRSchedule(post_warmup_learning_rate=1e-4, warmup_steps=num_warmup_steps)

# Compile the model
caption_model.compile(optimizer=keras.optimizers.Adam(lr_schedule), loss=cross_entropy)

# Fit the model
caption_model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=valid_dataset,
    callbacks=[early_stopping],
)

caption_model.save_weights(WEIGHTS_PATH)

caption_model = load_model(WEIGHTS_PATH, IMAGE_SIZE, BATCH_SIZE, EMBED_DIM,
    FF_DIM, SEQ_LENGTH, VOCAB_SIZE, train_data=train_data,
    process_input=process_input, image_augmentation=image_augmentation
)

vocab = vectorization.get_vocabulary()
index_lookup = dict(zip(range(len(vocab)), vocab))
max_decoded_sentence_length = SEQ_LENGTH - 1
valid_images = list(valid_data.keys())

# Check predictions for a few samples
generate_caption(caption_model, IMAGE_SIZE, valid_images, valid_data,
                 max_decoded_sentence_length, vectorization,
                 index_lookup)
generate_caption(caption_model, IMAGE_SIZE, valid_images, valid_data,
                 max_decoded_sentence_length, vectorization,
                 index_lookup)
generate_caption(caption_model, IMAGE_SIZE, valid_images, valid_data,
                 max_decoded_sentence_length, vectorization,
                 index_lookup)

