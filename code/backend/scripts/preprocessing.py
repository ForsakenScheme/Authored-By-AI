from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion
from sklearn.feature_selection import SelectKBest, chi2

from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.probability import FreqDist
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer

from spellchecker import SpellChecker

from backend.utils.configurating import config
from backend.utils.log import get_logger

from time import sleep

import re
import nltk
import string
import numpy as np
import requests
import spacy

try:
    wordnet.ensure_loaded()
except LookupError:
    nltk.download("wordnet")

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.pos_tag(["test"])
except LookupError:
    nltk.download("averaged_perceptron_tagger")

try:
    nltk.ne_chunk(nltk.pos_tag(["test"]))
except LookupError:
    nltk.download("maxent_ne_chunker")
    nltk.download("words")
    
logger = get_logger()

def split_data(X, y):
    """
    Split the data into training, validation, and testing sets.

    Parameters:
        X (array-like): The input features.
        y (array-like): The target variable.

    Returns:
        tuple: A tuple containing the training, validation, and testing sets for both X and y.
    """
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

######### Before preprocessing #########

class StopWords(BaseEstimator, TransformerMixin):
    """
    Custom transformer to calculate the ratio of stopwords to useful words in a text.

    Parameters:
        data_language (str): Language for stopwords (default is "english").
    
    Attributes:
        data_language (str): Language used for stopwords.
        stopwords (list): List of stopwords, loaded when required.
    """

    def __init__(self, data_language="english"):
        self.data_language = data_language
        self.stopwords = None  

    def _load_stopwords(self):
        """
        Lazy load stopwords to make the transformer pickleable.
        """
        if self.stopwords is None:
            try:
                self.stopwords = stopwords.words(self.data_language)
            except LookupError:
                nltk.download("stopwords")
                self.stopwords = stopwords.words(self.data_language)

    def fit(self, X, y=None):
        """
        Fit method for the StopWords transformer.

        Parameters:
        - X (array-like): Input data. Here, it's assumed to be a list of texts.
        - y (array-like): Target labels. Not used in this transformer.
        """
        return self

    def transform(self, X):
        """
        Transform method for the StopWords transformer.

        Parameters:
        - X (array-like): Input data. Here, it's assumed to be a list of texts.

        Returns:
        - stopword_ratio (numpy.ndarray): Array containing the ratio of stopwords to useful words for each text.
        """
        logger.info("Extracting stopword features..")
        self._load_stopwords()  

        stopword_ratio = []
        for text in X:
            words = text.split()
            useful_words_count = len(
                [word for word in words if word.lower() not in self.stopwords]
            )
            # Avoid division by zero in case of empty texts
            ratio = len(words) / useful_words_count if useful_words_count > 0 else 0
            stopword_ratio.append(ratio)

        stopword_ratio_features = np.array(stopword_ratio).reshape(-1, 1)
        logger.info(f"Shape of stopword features: {stopword_ratio_features.shape}")
        return stopword_ratio_features

class ErrorDetector(BaseEstimator, TransformerMixin):
    """
    Detects errors in text using pyspellchecker and LanguageTool API.

    Parameters:
        data_language (str): Language for error detection (default is "english").

    Attributes:
        data_language (str): The ISO code for the language.
        languagetool_code (str): Language code used for the LanguageTool API.
    """

    LANGUAGE_MAPPING = {
        "english": ("en", "en-US"),
        "german": ("de", "de-DE"),
        "french": ("fr", "fr-FR"),
    }

    def __init__(self, data_language="english"):
        if data_language not in self.LANGUAGE_MAPPING:
            raise ValueError(
                f"Language '{data_language}' not supported. Available languages: {', '.join(self.LANGUAGE_MAPPING.keys())}."
            )
        
        self.data_language, self.languagetool_code = self.LANGUAGE_MAPPING[data_language]
        self._spell_checker = None  

    def _load_spell_checker(self):
        """
        Lazy load the spell checker to make the transformer pickleable.
        """
        if self._spell_checker is None:
            self._spell_checker = SpellChecker(self.data_language)

    def fit(self, X, y=None):
        """
        Fit the ErrorDetector.

        Parameters:
            X (array-like): The input features.
            y (array-like): The target variable.

        Returns:
            self (ErrorDetector): The fitted ErrorDetector object.
        """
        # No training needed for spell checker and LanguageTool API
        return self

    def transform(self, X):
        """
        Transform the input text by extracting error features.

        Parameters:
            X (array-like): The input text data.

        Returns:
            numpy.ndarray: The error features.
        """
        logger.info("Extracting error features..")
        self._load_spell_checker()  

        spell_error_features = []
        grammar_error_features = []

        for text in X:
            # Spell checking
            words = text.split()
            misspelled = self._spell_checker.unknown(words)
            spell_error_count = len(misspelled)
            spell_error_features.append([spell_error_count])

            # Grammar checking
            grammar_error_count = self.check_grammar(text)
            grammar_error_features.append([grammar_error_count])

        spell_error_features = np.array(spell_error_features)
        grammar_error_features = np.array(grammar_error_features)
        error_features = np.hstack((spell_error_features, grammar_error_features))
        logger.info(f"Shape of error features: {error_features.shape}")
        return error_features

    def check_grammar(self, text):
        """
        Check the grammar of a text using the LanguageTool API.

        Parameters:
            text (str): The input text to check.

        Returns:
            num_errors (int): The number of grammar errors in the text.
        """
        # LanguageTool API endpoint
        url = "https://languagetool.org/api/v2/check"

        # Parameters
        params = {"language": self.languagetool_code, "text": text}

        # Retry parameters
        max_retries = 10
        retry_delay = 3

        # Retry connection to the API
        for retry in range(max_retries):
            try:
                response = requests.post(url, data=params)
                if response.status_code == 200:
                    # Parse the JSON response
                    data = response.json()
                    # Extract the number of grammar errors
                    num_errors = len(data["matches"])
                    return num_errors
                else:
                    logger.error(
                        f"Error during API call to Language Tool: {response.status_code} with request {response.request.url} and response {response.text}"
                    )
                    return 0
            except requests.exceptions.RequestException as e:
                logger.error(f"Error: {e}")
                if retry < max_retries - 1:
                    logger.info(f"Retrying connection to API in {retry_delay} seconds...")
                    sleep(retry_delay)
                else:
                    raise Exception(f"API couldn't be reached after {max_retries} retries.")

class PunctuationFrequency(BaseEstimator, TransformerMixin):
    """
    Extracts punctuation frequency features from text.

    Attributes:
        vectorizer (DictVectorizer): The DictVectorizer object (lazy-loaded).

    Methods:
        fit: Fit the PunctuationFrequency.
        transform: Transform the input text by extracting punctuation frequency features.
        analyze_punctuation: Analyze the punctuation frequency in a text.
    """

    def __init__(self):
        self._vectorizer = None

    def _load_vectorizer(self):
        """
        Lazy load the vectorizer to make the transformer pickleable.
        """
        if self._vectorizer is None:
            self._vectorizer = DictVectorizer()

    def fit(self, X, y=None):
        """
        Fit the PunctuationFrequency.

        Parameters:
            X (array-like): The input features.
            y (array-like): The target variable.

        Returns:
            self (PunctuationFrequency): The fitted PunctuationFrequency object.
        """
        self._load_vectorizer()
        self._vectorizer.fit([self.analyze_punctuation(text) for text in X])
        return self

    def transform(self, X):
        """
        Transform the input text by extracting punctuation frequency features.

        Parameters:
            X (array-like): The input text data.

        Returns:
            scipy.sparse.csr_matrix: The punctuation frequency features.
        """
        logger.info("Extracting punctuation features..")
        self._load_vectorizer()
        punctuation_frequency = [self.analyze_punctuation(text) for text in X]
        features = self._vectorizer.transform(punctuation_frequency)
        logger.info(f"Shape of punctuation features: {features.shape}")
        return features

    def analyze_punctuation(self, text):
        """
        Analyze the punctuation frequency in a text.

        Parameters:
            text (str): The input text.

        Returns:
            dict: The punctuation frequency.
        """
        punctuation_counts = {punct: 0 for punct in string.punctuation}

        for char in text:
            if char in string.punctuation:
                punctuation_counts[char] += 1

        total_chars = sum(punctuation_counts.values())
        if total_chars > 0:
            punctuation_frequency = {
                punct: count / total_chars
                for punct, count in punctuation_counts.items()
            }
        else:
            punctuation_frequency = punctuation_counts

        return punctuation_frequency

class SentenceLength(BaseEstimator, TransformerMixin):
    """
    Extracts sentence length features from text.

    Methods:
        fit: Fit the SentenceLength.
        transform: Transform the input text by extracting sentence length features.
        calculate_mean_sentence_length: Calculate the mean sentence length of a text.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        """
        Fit the SentenceLength.

        Parameters:
            X (array-like): The input features.
            y (array-like): The target variable.

        Returns:
            self (SentenceLength): The fitted SentenceLength object.
        """
        # No fitting logic needed, as this transformer is stateless.
        return self

    def transform(self, X):
        """
        Transform the input text by extracting sentence length features.

        Parameters:
            X (array-like): The input text data.

        Returns:
            numpy.ndarray: The sentence length features.
        """
        logger.info("Extracting sentence length features..")
        sentence_lengths = [self.calculate_mean_sentence_length(text) for text in X]
        sentence_lengths_features = np.array(sentence_lengths).reshape(-1, 1)
        logger.info(
            f"Shape of sentence length features: {sentence_lengths_features.shape}"
        )
        return sentence_lengths_features

    def calculate_mean_sentence_length(self, text):
        """
        Calculate the mean sentence length of a text.

        Parameters:
            text (str): The input text.

        Returns:
            float: The mean sentence length.
        """
        # Tokenize the text into sentences
        sentences = sent_tokenize(text)

        # Calculate the number of words in each sentence
        words_per_sentence = [len(word_tokenize(sentence)) for sentence in sentences]

        # Calculate the mean number of words per sentence
        if words_per_sentence:
            mean_words_per_sentence = sum(words_per_sentence) / len(words_per_sentence)
        else:
            mean_words_per_sentence = 0
        return mean_words_per_sentence

class NamedEntity(BaseEstimator, TransformerMixin):
    """
    Extracts named entity features from text.

    Attributes:
        vectorizer (DictVectorizer): The DictVectorizer object.
        nlp (Language): The spaCy language model, loaded lazily.

    Methods:
        fit: Fit the NamedEntity.
        transform: Transform the input text by extracting named entity features.
        extract_named_entities: Extract named entities from text.
        _load_spacy_model: Load the spaCy model only when needed.
    """

    def __init__(self, data_language="english"):
        self.vectorizer = None
        self.data_language = data_language
        self.nlp = None 

    def _load_spacy_model(self):
        """
        Load the appropriate spaCy model based on the language if not already loaded.

        Returns:
            Language: The loaded spaCy language model.
        """
        if self.nlp is None:
            language_map = {
                "english": "en_core_web_sm",
                "german": "de_core_news_sm",
                "french": "fr_core_news_sm"
            }
            model_name = language_map.get(self.data_language)
            if model_name is None:
                raise ValueError("Language not supported in spaCy models.")
            
            try:
                self.nlp = spacy.load(model_name)
            except OSError:
                raise OSError(f"Model '{model_name}' not found. Please download it using 'python -m spacy download {model_name}'.")

        return self.nlp

    def fit(self, X, y=None):
        """
        Fit the NamedEntity.

        Parameters:
            X (array-like): The input features.
            y (array-like): The target variable.

        Returns:
            self (NamedEntity): The fitted NamedEntity object.
        """
        named_entity_counts = [self.extract_named_entities(text) for text in X]
        self.vectorizer = DictVectorizer()
        self.vectorizer.fit(named_entity_counts)
        return self

    def transform(self, X):
        """
        Transform the input text by extracting named entity features.

        Parameters:
            X (array-like): The input text data.

        Returns:
            scipy.sparse.csr_matrix: The named entity features.
        """
        named_entity_counts = [self.extract_named_entities(text) for text in X]
        
        if self.vectorizer is None:
            raise ValueError("The vectorizer has not been fitted. Please call 'fit' before 'transform'.")

        logger.info("Extracting named entity features..")
        return self.vectorizer.transform(named_entity_counts)

    def extract_named_entities(self, text):
        """
        Extract named entities from text.

        Parameters:
            text (str): The input text.

        Returns:
            dict: The named entity counts.
        """
        ne_counts = {"ORG": 0, "PERSON": 0, "LOCATION": 0}
        nlp = self._load_spacy_model()
        doc = nlp(text)
        
        for ent in doc.ents:
            if ent.label_ == "ORG":
                ne_counts["ORG"] += 1
            elif ent.label_ == "PERSON":
                ne_counts["PERSON"] += 1
            elif ent.label_ == "GPE":
                ne_counts["LOCATION"] += 1
                
        return ne_counts

class SentimentAnalysis(BaseEstimator, TransformerMixin):
    """
    Extracts sentiment features from text.

    Attributes:
        vectorizer (DictVectorizer): The DictVectorizer object, initialized lazily.
        nlp (Language): The spaCy language model, loaded lazily.

    Methods:
        fit: Fit the SentimentAnalysis.
        transform: Transform the input text by extracting sentiment features.
        extract_sentiment_adjectives: Extract sentiment adjectives from text.
        extract_sentiment_adverbs: Extract sentiment adverbs from text.
        is_sentiment_adjective: Check if a word is a sentiment adjective.
        is_sentiment_adverb: Check if a word is a sentiment adverb.
    """

    def __init__(self, data_language="english"):
        self.vectorizer = None 
        self.data_language = data_language
        self.nlp = None 

    def _load_spacy_model(self):
        """
        Load the appropriate spaCy model based on the language if not already loaded.

        Returns:
            Language: The loaded spaCy language model.
        """
        if self.nlp is None:
            language_map = {
                "english": "en_core_web_sm",
                "german": "de_core_news_sm",
                "french": "fr_core_news_sm"
            }

            model_name = language_map.get(self.data_language)
            if model_name is None:
                raise ValueError("Language not supported. Choose 'english', 'german', or 'french'.")
            
            try:
                self.nlp = spacy.load(model_name)
            except OSError:
                raise OSError(f"Model '{model_name}' not found. Please download it using 'python -m spacy download {model_name}'.")

        return self.nlp

    def fit(self, X, y=None):
        """
        Fit the SentimentAnalysis.

        Parameters:
            X (array-like): The input features.
            y (array-like): The target variable.

        Returns:
            self (SentimentAnalysis): The fitted SentimentAnalysis object.
        """
        sentiment_features = [self.extract_sentiment_features(text) for text in X]
        self.vectorizer = DictVectorizer()
        self.vectorizer.fit(sentiment_features)
        return self

    def transform(self, X):
        """
        Transform the input text by extracting sentiment features.

        Parameters:
            X (array-like): The input text data.

        Returns:
            scipy.sparse.csr_matrix: The sentiment features.
        """
        sentiment_features = [self.extract_sentiment_features(text) for text in X]

        if self.vectorizer is None:
            raise ValueError("The vectorizer has not been fitted. Please call 'fit' before 'transform'.")

        logger.info("Extracting sentiment features..")
        return self.vectorizer.transform(sentiment_features)

    def extract_sentiment_features(self, text):
        """
        Extract sentiment features from text.

        Parameters:
            text (str): The input text.

        Returns:
            dict: The sentiment features with counts of adjectives and adverbs.
        """
        nlp = self._load_spacy_model() 
        doc = nlp(text)
        sentiment_adjectives = self.extract_sentiment_adjectives(doc)
        sentiment_adverbs = self.extract_sentiment_adverbs(doc)
        features = {
            "sentiment_adjectives_count": len(sentiment_adjectives),
            "sentiment_adverbs_count": len(sentiment_adverbs),
        }
        return features

    def extract_sentiment_adjectives(self, doc):
        """
        Extract sentiment adjectives from text.

        Parameters:
            doc (spacy.tokens.Doc): The spaCy document.

        Returns:
            list: The sentiment adjectives.
        """
        return [token.text for token in doc if self.is_sentiment_adjective(token)]

    def extract_sentiment_adverbs(self, doc):
        """
        Extract sentiment adverbs from text.

        Parameters:
            doc (spacy.tokens.Doc): The spaCy document.

        Returns:
            list: The sentiment adverbs.
        """
        return [token.text for token in doc if self.is_sentiment_adverb(token)]

    def is_sentiment_adjective(self, token):
        """
        Check if a token is a sentiment adjective.

        Parameters:
            token (spacy.tokens.Token): The token to check.

        Returns:
            bool: True if the token is a sentiment adjective, False otherwise.
        """
        return token.pos_ == "ADJ"

    def is_sentiment_adverb(self, token):
        """
        Check if a token is a sentiment adverb.

        Parameters:
            token (spacy.tokens.Token): The token to check.

        Returns:
            bool: True if the token is a sentiment adverb, False otherwise.
        """
        return token.pos_ == "ADV"
    
######### Preprocessing #########

class Preprocessing(BaseEstimator, TransformerMixin):
    """
    Preprocesses text data by removing punctuation and applying stemming or lemmatization.

    Parameters:
        punctuation (bool): Whether to remove punctuation from the text.
        stem (bool): Whether to apply stemming to the words.
        lemmatize (bool): Whether to apply lemmatization to the words.
        data_language (str): Language for stemming and lemmatization.
    """

    def __init__(self, punctuation=False, stem=False, lemmatize=False, data_language="english"):
        self.data_language = data_language
        self.punctuation = punctuation

        if stem and lemmatize:
            raise ValueError("Both stem and lemmatize cannot be True at the same time to avoid redundancy.")
        
        self.stem = stem
        self.lemmatize = lemmatize
        self.stemmer = None
        self.lemmatizer = None
        self.nlp = None

    def load_spacy_model(self, data_language):
        """Loads the appropriate spaCy model based on the specified language."""
        logger.info("Configuring spaCy model..")
        model_mapping = {
            "german": "de_core_news_sm",
            "french": "fr_core_news_sm",
        }

        if data_language not in model_mapping:
            logger.error("Language not supported in spaCy models.")
            raise ValueError("Language not supported in spaCy models.")

        model_name = model_mapping[data_language]

        try:
            logger.info(f"Loading spaCy model: {model_name}")
            nlp = spacy.load(model_name)
            logger.info(f"spaCy model for {data_language} loaded successfully.")
            return nlp
        except OSError:
            logger.error(f"Please download the {data_language.capitalize()} model using 'python -m spacy download {model_name}'.")
            raise OSError(f"Please download the {data_language.capitalize()} model using 'python -m spacy download {model_name}'.")

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        logger.info("Preprocessing..")
        if not (self.punctuation or self.stem or self.lemmatize):
            logger.info("No preprocessing applied..")
            return X

        # Load stemmer and lemmatizer only when needed
        if self.stem and not self.stemmer:
            self.stemmer = SnowballStemmer(language=self.data_language)
        if self.lemmatize and not self.lemmatizer and self.data_language == "english":
            self.lemmatizer = WordNetLemmatizer()
        if self.lemmatize and self.data_language != "english" and not self.nlp:
            self.nlp = self.load_spacy_model(self.data_language)

        Xt = [self.preprocess_text(text) for text in X]
        logger.info("Preprocessing applied..")
        return Xt

    def preprocess_text(self, text):
        pattern = r"[.,'\"!@#$%^&*(){}?/;`~:\]\[-]" if (self.punctuation or self.stem or self.lemmatize) else ""
        text_without_punctuation = re.sub(pattern, "", text)
        stops = set(stopwords.words("english"))
        words = text_without_punctuation.split()
        filtered_words = []

        if self.stem:
            for word in words:
                if word.lower() not in stops:
                    filtered_words.append(self.stemmer.stem(word))
        elif self.lemmatize:
            for word in words:
                if self.data_language != "english":
                    token = self.nlp(word)[0]
                    filtered_words.append(token.lemma_)
                else:
                    if word.lower() not in stops:
                        filtered_words.append(self.lemmatizer.lemmatize(word))
        else:
            for word in words:
                if word.lower() not in stops:
                    filtered_words.append(word)

        return " ".join(filtered_words)

######### After preprocessing #########

class WordLength(BaseEstimator, TransformerMixin):
    """
    Extracts word length features from text.

    Methods:
        fit: Fit the WordLength.
        transform: Transform the input text by extracting word length features.
        calculate_mean_word_length: Calculate the mean word length of a text.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        """
        Fit the WordLength.

        Parameters:
            X (array-like): The input features.
            y (array-like): The target variable.

        Returns:
            self (WordLength): The fitted WordLength object.
        """
        return self

    def transform(self, X):
        """
        Transform the input text by extracting word length features.

        Parameters:
            X (array-like): The input text data.

        Returns:
            numpy.ndarray: The word length features.
        """
        logger.info("Extracting word length features..")
        word_lengths = [self.calculate_mean_word_length(text) for text in X]
        word_lengths_features = np.array(word_lengths).reshape(-1, 1)
        logger.info(f"Shape of word length features: {word_lengths_features.shape}")
        return word_lengths_features

    def calculate_mean_word_length(self, text):
        """
        Calculate the mean word length of a text.

        Parameters:
            text (str): The input text.

        Returns:
            float: The mean word length.
        """
        # Tokenize the text into words
        words = word_tokenize(text)

        # Calculate the length of each word
        word_lengths = [len(word) for word in words]

        # Calculate the mean word length
        if word_lengths:
            mean_word_length = sum(word_lengths) / len(word_lengths)
        else:
            mean_word_length = 0
        return mean_word_length

class TextWordCounter(BaseEstimator, TransformerMixin):
    """
    Counts words in text data and extracts either frequency distribution features or TF-IDF features.

    Attributes:
        freqDict (bool): Whether to extract frequency distribution features.
        bigrams (bool): Whether to include bigrams in the frequency distribution features.
        dict_vectorizer (DictVectorizer): The DictVectorizer object, initialized lazily.
        tfidf_vectorizer (TfidfVectorizer): The TfidfVectorizer object, initialized lazily.

    Methods:
        fit: Fit the TextWordCounter.
        transform: Transform the input text by extracting either frequency distribution features or TF-IDF features.
        count_words: Count words in a text.
    """

    def __init__(self, freqDict=False, bigrams=False):
        self.freqDict = freqDict
        self.bigrams = bigrams
        self._dict_vectorizer = None  
        self._tfidf_vectorizer = None  

    @property
    def dict_vectorizer(self):
        """Lazy load the DictVectorizer."""
        if self._dict_vectorizer is None:
            self._dict_vectorizer = DictVectorizer()
        return self._dict_vectorizer

    @property
    def tfidf_vectorizer(self):
        """Lazy load the TfidfVectorizer."""
        if self._tfidf_vectorizer is None:
            self._tfidf_vectorizer = TfidfVectorizer()
        return self._tfidf_vectorizer

    def fit(self, X, y=None):
        if self.freqDict:
            # Fit the DictVectorizer on the word frequency distributions
            fdist_list = [self.count_words(text) for text in X]
            self.dict_vectorizer.fit(fdist_list)
        else:
            # Fit the TfidfVectorizer on the text data
            self.tfidf_vectorizer.fit(X)
        return self

    def transform(self, X):
        if self.freqDict:
            logger.info("Extracting freqDict features..")
            fdist_list = [self.count_words(text) for text in X]
            features = self.dict_vectorizer.transform(fdist_list)
            logger.info(f"Shape of freqDict features: {features.shape}")
            return features
        else:
            logger.info("Extracting tfidf features..")
            features = self.tfidf_vectorizer.transform(X)
            logger.info(f"Shape of tfidf features: {features.shape}")
            return features

    def count_words(self, text):
        words = word_tokenize(text)
        if self.bigrams:
            fdist = FreqDist(words)
            bigrams = [f"{words[i]} {words[i + 1]}" for i in range(len(words) - 1)]
            for bigram in bigrams:
                bigram_lower = bigram.lower()
                if bigram_lower in fdist:
                    fdist[bigram_lower] += 1
                else:
                    fdist[bigram_lower] = 1
        else:
            fdist = FreqDist(words)
        return fdist

class VocabularySize(BaseEstimator, TransformerMixin):
    """
    Extracts vocabulary size features from text.

    Methods:
        fit: Fit the VocabularySize.
        transform: Transform the input text by extracting vocabulary size features.
        calculate_vocab_size: Calculate the vocabulary size of a text.
    """

    def fit(self, X, y=None):
        """
        Fit the VocabularySize.

        Parameters:
            X (array-like): The input features.
            y (array-like): The target variable.

        Returns:
            self (VocabularySize): The fitted VocabularySize object.
        """
        return self

    def transform(self, X):
        """
        Transform the input text by extracting vocabulary size features.

        Parameters:
            X (array-like): The input text data.

        Returns:
            numpy.ndarray: The vocabulary size features.
        """
        logger.info("Extracting vocabulary size features..")
        vocab_sizes = [self.calculate_vocab_size(text) for text in X]
        vocab_sizes_features = np.array(vocab_sizes).reshape(-1, 1)
        logger.info(f"Shape of vocabulary size features: {vocab_sizes_features.shape}")
        return vocab_sizes_features

    def calculate_vocab_size(self, text):
        """
        Calculate the vocabulary size of a text.

        Parameters:
            text (str): The input text.

        Returns:
            int: The vocabulary size.
        """
        unique_words = set(text.split())
        return len(unique_words)

######### Feature Extractors #########

class FeatureExtractorBeforePreprocessing(BaseEstimator, TransformerMixin):
    """
    Extract the FeatureExtractorBeforePreprocessing.

    Attributes:
        stopWords (StopWords): The StopWords object.
        errorDetector (ErrorDetector): The ErrorDetector object.
        punctuationFrequency (PunctuationFrequency): The PunctuationFrequency object.
        sentenceLength (SentenceLength): The SentenceLength object.
        namedEntity (NamedEntity): The NamedEntity object.
        sentimentAnalysis (SentimentAnalysis): The SentimentAnalysis object.
        feature_union (list): List of feature transformers.
        combined_transformers (FeatureUnion): The combined transformers.

    Parameters:
        stopWords (bool): Whether to include stop words as a feature.
        errorDetector (bool): Whether to include error detection as a feature.
        punctuationFrequency (bool): Whether to include punctuation frequency as a feature.
        sentenceLength (bool): Whether to include sentence length as a feature.
        namedEntity (bool): Whether to include named entity as a feature.
        sentimentAnalysis (bool): Whether to include sentiment analysis as a feature.

    Methods:
        fit: Fit the FeatureExtractorBeforePreprocessing object.
        transform: Transform the input data by extracting the specified features before preprocessing.
    """

    def __init__(
        self,
        stopWords=False,
        errorDetector=False,
        punctuationFrequency=False,
        sentenceLength=False,
        namedEntity=False,
        sentimentAnalysis=False,
        data_language = "english"
    ):
        self.stopWords = stopWords
        self.errorDetector = errorDetector
        self.punctuationFrequency = punctuationFrequency
        self.sentenceLength = sentenceLength
        self.namedEntity = namedEntity
        self.sentimentAnalysis = sentimentAnalysis
        self.feature_union = []
        self.combined_transformers = None
        self.data_language = data_language
        logger.info("==================================================================================")
        logger.info("Initialized FeatureExtractorBeforePreprocessing object..")

        # Add transformers based on selected options
        if self.stopWords:
            logger.info("Adding StopWords transformer..")
            self.feature_union.append(("stopWords", StopWords(self.data_language)))
        if self.errorDetector:
            logger.info("Adding ErrorDetector transformer..")
            self.feature_union.append(("errorDetector", ErrorDetector(self.data_language)))
        if self.punctuationFrequency:
            logger.info("Adding PunctuationFrequency transformer..")
            self.feature_union.append(("punctuationFrequency", PunctuationFrequency()))
        if self.sentenceLength:
            logger.info("Adding SentenceLength transformer..")
            self.feature_union.append(("sentenceLength", SentenceLength()))
        if self.namedEntity:
            logger.info("Adding NamedEntity transformer..")
            self.feature_union.append(("namedEntity", NamedEntity(self.data_language)))
        if self.sentimentAnalysis:
            logger.info("Adding SentimentAnalysis transformer..")
            self.feature_union.append(("sentimentAnalysis", SentimentAnalysis(self.data_language)))
        if self.feature_union:
            logger.info("Creating combined transformers..")
            self.combined_transformers = FeatureUnion(self.feature_union)
        logger.info("Added all FeatureExtractorBeforePreprocessing transformers based on selected options..")
        
    def fit(self, X, y=None):
        """
        Fit the FeatureExtractorBeforePreprocessing.

        Parameters:
            X (array-like): The input features.
            y (array-like): The target variable.

        Returns:
            self (FeatureExtractorBeforePreprocessing): The fitted FeatureExtractorBeforePreprocessing object.
        """
        # Fit the combined transformers if they exist
        if self.combined_transformers:
            self.combined_transformers.fit(X)
        return self

    def transform(self, X):
        """
        Transform the input data using the combined transformers.

        Parameters:
            X (array-like): The input features.

        Returns:
            X_array (numpy.ndarray): The transformed input data before preprocessing.
        """
        # Transform the input data using the combined transformers
        if self.feature_union:
            logger.info("Extracting features before preprocessing..")
            combined_features = self.combined_transformers.transform(X)
            logger.info(
                f"Shape of combined features before preprocessing: {combined_features.shape}"
            )
            return combined_features
        else:
            X_array = np.empty((len(X), 0))
            logger.info("Nothing to extract before preprocessing..")
            logger.info(
                f"Shape of combined features before preprocessing: {X_array.shape}"
            )
            return X_array
        
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            "stopWords": self.stopWords,
            "errorDetector": self.errorDetector,
            "punctuationFrequency": self.punctuationFrequency,
            "sentenceLength": self.sentenceLength,
            "namedEntity": self.namedEntity,
            "sentimentAnalysis": self.sentimentAnalysis,
            "data_language": self.data_language,
        }

class FeatureExtractorAfterPreprocessing(BaseEstimator, TransformerMixin):
    """
    Extracts features from text data after preprocessing.

    Attributes:
        config: Configuration object.
        textWordCounter (TextWordCounter): The TextWordCounter object.
        wordLength (WordLength): The WordLength object.
        vocabularySize (VocabularySize): The VocabularySize object.
        feature_union (list): List of feature transformers.
        combined_transformers (FeatureUnion): The combined transformers.

    Parameters:
        textWordCounter (bool): Whether to include text word counter as a feature.
        wordLength (bool): Whether to include word length as a feature.
        vocabularySize (bool): Whether to include vocabulary size as a feature.

    Methods:
        fit: Fit the FeatureExtractorAfterPreprocessing object.
        transform: Transform the input data by extracting the specified features after preprocessing.
    """

    def __init__(
        self,
        config=config,
        textWordCounter=False,
        wordLength=False,
        vocabularySize=False,
    ):
        self.config = config
        self.textWordCounter = textWordCounter
        self.wordLength = wordLength
        self.vocabularySize = vocabularySize
        self.feature_union = []
        logger.info("Initialized FeatureExtractorAfterPreprocessing object..")

        if self.textWordCounter:
            self.feature_union.append(
                (
                    "textWordCounter",
                    TextWordCounter(
                        self.config.getboolean("TextWordCounter", "freqDist"),
                        self.config.getboolean("TextWordCounter", "bigrams"),
                    ),
                )
            )

        if self.wordLength:
            logger.info("Adding WordLength transformer..")
            self.feature_union.append(("wordLength", WordLength()))

        if self.vocabularySize:
            logger.info("Adding VocabularySize transformer..")
            self.feature_union.append(("vocabularySize", VocabularySize()))

        self.combined_transformers = FeatureUnion(self.feature_union)
        logger.info("Added FeatureExtractorAfterPreprocessing transformers based on selected options..")

    def fit(self, X, y=None):
        """
        Fit the FeatureExtractorAfterPreprocessing object.

        Parameters:
            X (array-like): The input features.
            y (array-like): The target variable.

        Returns:
            self (FeatureExtractorAfterPreprocessing): The fitted FeatureExtractorAfterPreprocessing object.
        """
        if self.combined_transformers:
            self.combined_transformers.fit(X)
        return self

    def transform(self, X):
        """
        Transform the input data by extracting the specified features after preprocessing.

        Parameters:
            X (array-like): The input features.

        Returns:
            X_array (numpy.ndarray): The transformed input data after preprocessing.
        """
        if self.feature_union:
            logger.info("Extracting features after preprocessing..")
            combined_features = self.combined_transformers.transform(X)
            logger.info(
                f"Shape of combined features after preprocessing: {combined_features.shape}"
            )
            return combined_features
        else:
            X_array = np.empty((len(X), 0))
            logger.info("Nothing to extract before preprocessing..")
            logger.info(
                f"Shape of combined features after preprocessing: {X_array.shape}"
            )
            return X_array
    
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            "textWordCounter": self.textWordCounter,
            "wordLength": self.wordLength,
            "vocabularySize": self.vocabularySize,
        }

######### Feature Selection #########

class FeatureSelection(SelectKBest):
    """
    Selects the top k best features.

    Methods:
        fit: Fit the FeatureSelection object.
        transform: Transform the input data by selecting the top k best features.
    """

    def __init__(self, k="all", score_func=chi2):
        super().__init__(score_func=score_func, k=k)
        self.k_value = k

    def fit(self, X, y):
        """
        Fit the FeatureSelection object.

        Parameters:
            X (array-like): The input features.
            y (array-like): The target variable.

        Returns:
            self (FeatureSelection): The fitted FeatureSelection object.
        """
        # Adjust k_value dynamically based on a ratio extracted during training
        num_features_available = X.shape[1]
        if self.k_value == "all":
            self.k = "all"
        else:
            ratio = self.k_value / num_features_available
            if ratio > 1:
                self.k = "all"
                logger.info(f"Ratio is greater than 1. Setting k to 'all'.")
            else:
                self.k = int(np.ceil(num_features_available * ratio))
                logger.info(
                    f"Adjusted k to {self.k} based on the ratio {ratio} and the number of available features: {num_features_available}"
                )
        return super().fit(X, y)

    def transform(self, X):
        """
        Transform the input data by selecting the top k best features.

        Parameters:
            X (array-like): The input features.

        Returns:
            X_selected (numpy.ndarray): The transformed input data with selected features.
        """
        logger.info(f"Selecting top {self.k} features..")
        logger.info(f"Shape of X before feature selection: {X.shape}")
        logger.info(f"Selected {len(self.get_support())} features.")
        logger.info(
            f"Shape of X after feature selection: {X[:, self.get_support()].shape}"
        )
        return super().transform(X)
    
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            "k": self.k_value,
            "score_func": self.score_func,
        }
