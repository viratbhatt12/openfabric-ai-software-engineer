import os
import json

intents_path = os.getcwd() + "/intents/intents.json"
if not os.path.exists(intents_path):
    intents_path = os.getcwd() + "/model/intents/intents.json"


class IntentsManager:
    """
    IntentsManager class for managing intents data.
    """

    def __init__(self, intents_path_=intents_path):
        """
        Initializes IntentsManager with the path to the intents JSON file.

        Args:
            intents_path_ (str): Path to the intents JSON file.
        """
        try:
            with open(intents_path_, "r") as intents_file:
                self.__intents = json.load(intents_file)
                self.patterns, self.responses = self.get_patterns_and_responses()
        except FileNotFoundError:
            print(f"Intents file is not found in {intents_path_}")
            self.__intents = {}
            self.patterns, self.responses = [], []
        except Exception as e:
            raise RuntimeError(f"Error initializing IntentsManager: {e}")

    def get_intents(self):
        """
        Returns the loaded intents data.

        Returns:
            dict: Intents data.
        """
        try:
            return self.__intents
        except Exception as e:
            raise RuntimeError(f"Error retrieving intents: {e}")

    def get_patterns_and_responses(self):
        """
        Extracts patterns and responses from the loaded intents data.

        Returns:
            tuple: (patterns, responses), where patterns is a list of input patterns and responses is a list of corresponding responses.
        """
        try:
            patterns = []
            responses = []

            # Extract patterns and responses from intents
            for intent in self.__intents.get("intents", []):
                patterns.extend(intent.get("patterns", []))
                responses.extend(intent.get("responses", []))

            return patterns, responses
        except Exception as e:
            raise RuntimeError(f"Error extracting patterns and responses: {e}")

    def get_words_and_index(self):
        """
        Creates a set of unique words and a dictionary mapping words to integers.

        Returns:
            tuple: (words, word_index), where words is a set of unique words and word_index is a dictionary mapping words to integers.
        """
        try:
            # Create a set of unique words
            words = set([word.lower() for pattern in self.patterns for word in pattern.split()])

            # Create a dictionary to map words to integers
            word_index = {word: idx + 1 for idx, word in enumerate(sorted(list(words)))}
            return words, word_index
        except Exception as e:
            raise RuntimeError(f"Error creating words and index: {e}")


class DataProcessor:
    """
    DataProcessor class for processing intents data.
    """

    def __init__(self):
        """
        Initializes DataProcessor with an instance of IntentsManager and loads intents data.
        """
        try:
            self.intents_manager = IntentsManager()
            self.intents_data = self.intents_manager.get_intents()
            _, self.word_idx = self.intents_manager.get_words_and_index()
        except Exception as e:
            raise RuntimeError(f"Error initializing DataProcessor: {e}")

    def get_training_data(self):
        """
        Generates training data from loaded intents data.

        Returns:
            tuple: (x_train, y_train), where x_train is a list of input patterns and y_train is a list of corresponding labels.
        """
        x_train = []
        y_train = []

        try:
            # Create training data
            for idx, intent in enumerate(self.intents_data["intents"]):
                for pattern in intent["patterns"]:
                    x_train.append([self.word_idx[word.lower()] for word in pattern.split()])
                    y_train.append(idx)
        except Exception as e:
            raise RuntimeError(f"Error generating training data: {e}")

        return x_train, y_train

    def get_word_idx(self):
        """
        Returns the word index dictionary.

        Returns:
            dict: Word index dictionary.
        """
        return self.word_idx

    def get_intents_data(self):
        """
        Returns the loaded intents data.

        Returns:
            dict: Intents data.
        """
        return self.intents_data
