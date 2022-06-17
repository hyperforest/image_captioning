import numpy as np
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt

from dataset import decode_and_resize, make_dataset
from model import (
    get_cnn_model,
    ImageCaptioningModel,
    TransformerDecoderBlock,
    TransformerEncoderBlock,
)


def load_model(weights_path, image_size, batch_size, embed_dim, ff_dim, seq_length,
    vocab_size, train_data, process_input, image_augmentation):
    
    # build model
    cnn_model = get_cnn_model(image_size=image_size)
    encoder = TransformerEncoderBlock(embed_dim=embed_dim, dense_dim=ff_dim, num_heads=1)
    decoder = TransformerDecoderBlock(embed_dim=embed_dim, ff_dim=ff_dim, num_heads=2,
        seq_length=seq_length, vocab_size=vocab_size)
    caption_model = ImageCaptioningModel(
        cnn_model=cnn_model, encoder=encoder, decoder=decoder, image_aug=image_augmentation,
    )
    cross_entropy = keras.losses.SparseCategoricalCrossentropy(
        from_logits=False, reduction="none"
    )
    caption_model.compile(optimizer=keras.optimizers.Adam(), loss=cross_entropy)

    # call model once to determine input shape
    train_dataset = make_dataset(
        list(train_data.keys()),
        list(train_data.values()),
        batch_size, process_input
    )
    batch_data = next(iter(train_dataset))
    _ = caption_model(batch_data)
    
    # load weights
    caption_model.load_weights(weights_path)
    return caption_model


def generate_caption(caption_model, image_size, valid_images, valid_data,
                     max_decoded_sentence_length, vectorization,
                     index_lookup):
    # Select a random image from the validation dataset
    sample_img_name = np.random.choice(valid_images)

    # Read the image from the disk
    sample_img = decode_and_resize(sample_img_name, image_size)
    img = sample_img.numpy().clip(0, 255).astype(np.uint8)
    plt.imshow(img)
    plt.show()

    # Pass the image to the CNN
    img = tf.expand_dims(sample_img, 0)
    img = caption_model.cnn_model(img)

    # Pass the image features to the Transformer encoder
    encoded_img = caption_model.encoder(img, training=False)

    # Generate the caption using the Transformer decoder
    decoded_caption = "<start> "
    for i in range(max_decoded_sentence_length):
        tokenized_caption = vectorization([decoded_caption])[:, :-1]
        mask = tf.math.not_equal(tokenized_caption, 0)
        predictions = caption_model.decoder(
            tokenized_caption, encoded_img, training=False, mask=mask
        )
        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = index_lookup[sampled_token_index]
        if sampled_token == " <end>":
            break
        decoded_caption += " " + sampled_token

    decoded_caption = decoded_caption.replace("<start> ", "")
    decoded_caption = decoded_caption.replace(" <end>", "").strip()
    print("Predicted Caption:", decoded_caption)

    print("Actual Captions:")
    actual_captions = valid_data[sample_img_name]
    for caption in actual_captions:
        caption = caption.replace("<start> ", "")
        caption = caption.replace(" <end>", "").strip()
        print('>', caption)
