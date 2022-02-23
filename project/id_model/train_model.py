import os
from tensorflow import keras
from keras import Model, layers

# Run from repository root
DATA_DIR = "data/"

# Max dimensions for input images
MAX_X = 640
MAX_Y = 1136


def get_writers(data_file: str) -> tuple:
    sample2writer = {}
    writer2samples = {}
    with open(data_file, "r") as f:
        for line in f:
            sample_id = line.split(" ")[0]
            writer_id = line.split(" ")[1]
            sample2writer[sample_id] = writer_id
            if writer_id in writer2samples:
                writer2samples[writer_id].append(sample_id)
            else:
                writer2samples[writer_id] = [sample_id]

        return sample2writer, writer2samples


def gen_model(writer2samples: dict) -> Model:
    n_classes = len(writer2samples.keys())
    base_model = keras.applications.xception.Xception(
        weights="imagenet", include_top=False)
    fingerprint = layers.Dense(64, activation="relu")(base_model.output)
    dropout = layers.Dropout(0.5)(fingerprint)
    output = layers.Dense(n_classes)(dropout)
    model = Model(inputs=base_model.input, outputs=output)

    return model


if __name__ == "__main__":
    sample2writer, writer2samples = get_writers(
        os.path.join(DATA_DIR, "forms_for_parsing.txt"))
    model = gen_model(writer2samples)
