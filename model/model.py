import os
import random
import tensorflow as tf

utils_path = os.getcwd() + "/model/utils.py"
if not os.path.exists(utils_path):
    from utils import DataProcessor
else:
    from model.utils import DataProcessor

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D


class NLPModel:
    """
    NLPModel class for natural language processing using a TensorFlow model.
    """

    def __init__(self, model_path="model.h5", epochs=1000, verbose=2, confidence=0.5):
        """
        Initializes NLPModel.

        Args:
            model_path (str): Path to save/load the model.
            epochs (int): Number of epochs for training.
            verbose (int): Verbosity level during training.
        """
        try:
            self.data_processor = DataProcessor()
            self.X_train, self.Y_train = self.data_processor.get_training_data()
            self.intents_data = self.data_processor.get_intents_data()
            self.word_index = self.data_processor.get_word_idx()
            self.epochs = epochs
            self.model_path = model_path
            self.verbose = verbose
            self.max_length = self.get_patterns_length()
            self.model = None
            self.confidence_ = confidence
        except Exception as e:
            raise RuntimeError(f"Error initializing NLPModel: {e}")

    def build_model(self):
        """
        Builds and trains the NLP model.
        """
        try:
            # Convert X_train to a fixed size using padding
            x_train = tf.keras.preprocessing.sequence.pad_sequences(self.X_train, maxlen=self.max_length)

            # Convert y_train to one-hot encoding
            y_train = tf.keras.utils.to_categorical(self.Y_train, num_classes=len(self.intents_data["intents"]))

            # Build the model
            model = Sequential()
            model.add(Embedding(input_dim=len(self.word_index) + 1, output_dim=32, input_length=self.max_length))
            model.add(GlobalAveragePooling1D())
            model.add(Dense(len(self.intents_data["intents"]), activation="softmax"))

            # Compile the model
            model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

            # Train the model
            model.fit(x_train, y_train, epochs=self.epochs, verbose=self.verbose)

            # Save the model
            model.save(self.model_path)

            self.model = model

        except Exception as e:
            raise RuntimeError(f"Error building and training the model: {e}")

    def load_model(self, model_path_):
        """
        Loads a pre-trained model.
        """
        try:
            loaded_model = tf.keras.models.load_model(model_path_)
            self.model = loaded_model
        except Exception as e:
            raise RuntimeError(f"Error loading the model: {e}")

    def get_patterns_length(self):
        """
        Computes the maximum length of patterns in the training data.

        Returns:
            int: Maximum length of patterns.
        """
        try:
            max_length = max(len(pattern) for pattern in self.X_train)
            return max_length
        except Exception as e:
            raise RuntimeError(f"Error computing patterns length: {e}")

    def predict(self, input_text):
        """
        Get a response from the NLP model based on user input.

        Args:
            input_text (str): User input text.

        Returns:
            str: Generated response.
        """
        try:
            user_input = [self.word_index.get(word.lower(), 0) for word in input_text.split()]
            user_input = tf.keras.preprocessing.sequence.pad_sequences([user_input], maxlen=self.max_length)
            prediction = self.model.predict(user_input)
            intent_index = tf.argmax(prediction, axis=1).numpy()[0]
            confidence = prediction[0][intent_index]

            # If confidence is below a certain threshold, provide a generic response
            if confidence < self.confidence_:
                return "I'm not completely sure about that, but here's something relevant: " + random.choice(
                    self.intents_data["intents"][intent_index]['responses'])

            response = random.choice(self.intents_data["intents"][intent_index]['responses'])
            return response
        except Exception as e:
            raise RuntimeError(f"Error getting response: {e}")


if __name__ == '__main__':
    try:
        nlp_model = NLPModel()
        nlp_model.build_model()

    except Exception as e:
        print("Error initializing and training the NLP model:", str(e))
