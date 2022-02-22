import os
from tensorflow import keras

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


def gen_model(sample2writer: dict, writer2sample: dict) -> keras.Model:
    n_classes = len(writer2sample.keys())
    model = keras.models.Sequential([
        keras.layers.MaxPooling2D(2),
        keras.layers.Conv2D(128, 7, activation="relu", padding="same"),
        keras.layers.Conv2D(128, 7, activation="relu", padding="same"),
        keras.layers.MaxPooling2D(2),
        keras.layers.Conv2D(256, 5, activation="relu", padding="same"),
        keras.layers.Conv2D(256, 5, activation="relu", padding="same"),
        keras.layers.MaxPooling2D(2),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(25, activation="relu"),
        keras.layers.Dense(n_classes, activation="softmax"),
    ])


if __name__ == "__main__":
    sample2writer, writer2sample = get_writers(os.path.join(DATA_DIR, "forms_for_parsing.txt"))
    print(writer2sample)