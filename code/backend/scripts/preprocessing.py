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
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

from spellchecker import SpellChecker

from backend.utils.configurating import config
from backend.utils.log import setup_logging

from time import sleep

import re
import nltk
import string
import copy
import numpy as np
import requests

try:
    wordnet.ensure_loaded()
except LookupError:
    nltk.download("wordnet")

try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords")

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

logger = setup_logging("local")


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

    Attributes:
        stopwords_english (list): List of English stopwords.

    Methods
        fit: Fit method for the StopWords transformer.
        transform: Transform method for the StopWords transformer.
    """

    def __init__(self):
        self.stopwords_english = stopwords.words("english")

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
        stopword_ratio = []
        for text in X:
            words = text.split()
            useful_words_count = len(
                [word for word in words if word.lower() not in self.stopwords_english]
            )
            stopword_ratio.append(len(words) / useful_words_count)

        stopword_ratio_features = np.array(stopword_ratio).reshape(-1, 1)
        logger.info(f"Shape of stopword features: {stopword_ratio_features.shape}")
        return stopword_ratio_features


class ErrorDetector(BaseEstimator, TransformerMixin):
    """
    Detects errors in text using pyspellchecker and LanguageTool API.

    Attributes:
        spell_checker (SpellChecker): The SpellChecker object.

    Methods:
        fit: Fit the ErrorDetector.
        transform: Transform the input text by extracting error features.
        check_grammar: Check the grammar of a text using the LanguageTool API.
    """

    def __init__(self):
        self.spell_checker = SpellChecker()

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
        spell_error_features = []
        grammar_error_features = []
        for text in X:
            # Spell checking
            words = text.split()
            misspelled = self.spell_checker.unknown(words)
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
        params = {"language": "en-US", "text": text}

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
                    logger.info(
                        f"Retrying connection to API in {retry_delay} seconds..."
                    )
                    sleep(retry_delay)
                else:
                    raise Exception(
                        f"API couldn't be reached after {max_retries} retries."
                    )


class PunctuationFrequency(BaseEstimator, TransformerMixin):
    """
    Extracts punctuation frequency features from text.

    Attributes:
        vectorizer (DictVectorizer): The DictVectorizer object.

    Methods:
        fit: Fit the PunctuationFrequency.
        transform: Transform the input text by extracting punctuation frequency features.
        analyze_punctuation: Analyze the punctuation frequency in a text.
    """

    def __init__(self):
        self.vectorizer = DictVectorizer()

    def fit(self, X, y=None):
        """
        Fit the PunctuationFrequency.

        Parameters:
            X (array-like): The input features.
            y (array-like): The target variable.

        Returns:
            self (PunctuationFrequency): The fitted PunctuationFrequency object.
        """
        self.vectorizer.fit([self.analyze_punctuation(text) for text in X])
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
        punctuation_frequency = [self.analyze_punctuation(text) for text in X]
        features = self.vectorizer.transform(punctuation_frequency)
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
        # tokenize the text into sentences
        sentences = sent_tokenize(text)

        # calculate the number of words in each sentence
        words_per_sentence = [len(word_tokenize(sentence)) for sentence in sentences]

        # calculate the mean number of words per sentence
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

    Methods:
        fit: Fit the NamedEntity.
        transform: Transform the input text by extracting named entity features.
        extract_named_entities: Extract named entities from text.
    """

    def __init__(self):
        self.vectorizer = DictVectorizer()

    def fit(self, X, y=None):
        """
        Fit the NamedEntity.

        Parameters:
            X (array-like): The input features.
            y (array-like): The target variable.

        Returns:
            self (NamedEntity): The fitted NamedEntity object.
        """
        return self

    def transform(self, X):
        """
        Transform the input text by extracting named entity features.

        Parameters:
            X (array-like): The input text data.

        Returns:
            scipy.sparse.csr_matrix: The named entity features.
        """
        named_entity_counts = []
        for text in X:
            ne_counts = self.extract_named_entities(text)
            named_entity_counts.append(ne_counts)
        logger.info("Extracting named entity features..")
        return self.vectorizer.fit_transform(named_entity_counts)

    def extract_named_entities(self, text):
        """
        Extract named entities from text.

        Parameters:
            text (str): The input text.

        Returns:
            dict: The named entity counts.
        """
        ne_counts = {"ORG": 0, "PERSON": 0, "LOCATION": 0}
        tagged_text = nltk.pos_tag(nltk.word_tokenize(text))
        named_entities = nltk.ne_chunk(tagged_text)
        for entity in named_entities:
            if isinstance(entity, nltk.Tree):
                entity_type = entity.label()
                if entity_type == "ORGANIZATION":
                    ne_counts["ORG"] += 1
                elif entity_type == "PERSON":
                    ne_counts["PERSON"] += 1
                elif entity_type == "GPE":
                    ne_counts["LOCATION"] += 1
        return ne_counts


class SentimentAnalysis(BaseEstimator, TransformerMixin):
    """
    Extracts sentiment features from text.

    Attributes:
        vectorizer (DictVectorizer): The DictVectorizer object.

    Methods:
        fit: Fit the SentimentAnalysis.
        transform: Transform the input text by extracting sentiment features.
        extract_sentiment_adjectives: Extract sentiment adjectives from text.
        extract_sentiment_adverbs: Extract sentiment adverbs from text.
        is_sentiment_adjective: Check if a word is a sentiment adjective.
        is_sentiment_adverb: Check if a word is a sentiment adverb.
    """

    def __init__(self):
        self.vectorizer = DictVectorizer()

    def fit(self, X, y=None):
        """
        Fit the SentimentAnalysis.

        Parameters:
            X (array-like): The input features.
            y (array-like): The target variable.

        Returns:
            self (SentimentAnalysis): The fitted SentimentAnalysis object.
        """
        return self

    def transform(self, X):
        """
        Transform the input text by extracting sentiment features.

        Parameters:
            X (array-like): The input text data.

        Returns:
            scipy.sparse.csr_matrix: The sentiment features.
        """
        sentiment_features = []
        for text in X:
            tagged_text = nltk.pos_tag(nltk.word_tokenize(text))
            sentiment_adjectives = self.extract_sentiment_adjectives(tagged_text)
            sentiment_adverbs = self.extract_sentiment_adverbs(tagged_text)
            features = {
                "sentiment_adjectives_count": len(sentiment_adjectives),
                "sentiment_adverbs_count": len(sentiment_adverbs),
            }
            sentiment_features.append(features)
        logger.info("Extracting sentiment features..")
        return self.vectorizer.fit_transform(sentiment_features)

    def extract_sentiment_adjectives(self, tagged_text):
        """
        Extract sentiment adjectives from text.

        Parameters:
            tagged_text (list): The tagged text.

        Returns:
            list: The sentiment adjectives.
        """
        sentiment_adjectives = [
            word for word, tag in tagged_text if self.is_sentiment_adjective(tag)
        ]
        return sentiment_adjectives

    def extract_sentiment_adverbs(self, tagged_text):
        """
        Extract sentiment adverbs from text.

        Parameters:
            tagged_text (list): The tagged text.

        Returns:
            list: The sentiment adverbs.
        """
        sentiment_adverbs = [
            word for word, tag in tagged_text if self.is_sentiment_adverb(tag)
        ]
        return sentiment_adverbs

    def is_sentiment_adjective(self, tag):
        """
        Check if a word is a sentiment adjective.

        Parameters:
            tag (str): The word tag.

        Returns:
            bool: True if the word is a sentiment adjective, False otherwise.
        """
        return str(tag).startswith("JJ")

    def is_sentiment_adverb(self, tag):
        """
        Check if a word is a sentiment adverb.

        Parameters:
            tag (str): The word tag.

        Returns:
            bool: True if the word is a sentiment adverb, False otherwise.
        """
        return str(tag).startswith("RB")


######### Preprocessing #########


class Preprocessing(BaseEstimator, TransformerMixin):
    """
    Preprocesses text data by removing punctuation and applying stemming or lemmatization.

    Attributes:
        stemmer (PorterStemmer): The PorterStemmer object.
        lemmatizer (WordNetLemmatizer): The WordNetLemmatizer object.

    Parameters:
        punctuation (bool): Whether to remove punctuation from the text.
        stem (bool): Whether to apply stemming to the words.
        lemmatize (bool): Whether to apply lemmatization to the words.

    Methods:
        fit: Fit the TextPreprocessor.
        transform: Transform the input text by applying the specified preprocessing steps.
        preprocess_text: Preprocess a single text by removing punctuation and applying stemming or lemmatization.
    """

    def __init__(self, punctuation=False, stem=False, lemmatize=False):
        self.punctuation = punctuation
        if stem and lemmatize:
            raise ValueError(
                "Both stem and lemmatize cannot be True at the same time to avoid redundancy."
            )
        self.stem = stem
        self.lemmatize = lemmatize
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()

    def fit(self, X, y=None):
        """
        Fit the TextPreprocessor.

        Parameters:
            X (array-like): The input features.
            y (array-like): The target variable.

        Returns:
            self (TextPreprocessor): The fitted TextPreprocessor object.
        """
        return self

    def transform(self, X):
        """
        Transform the input text by applying the specified preprocessing steps.

        Parameters:
            X (array-like): The input text data.

        Returns:
            array-like: The preprocessed text data.
        """
        logger.info("Preprocessing..")
        if self.punctuation == False and self.stem == False and self.lemmatize == False:
            logger.info("No preprocessing applied..")
            logger.info(f"Returned list X of {len(X)} texts.")
            return X

        else:
            Xt = [self.preprocess_text(text) for text in X]
            logger.info("Preprocessing applied..")
            logger.info(f"Returned list Xt of {len(Xt)} preprocessed texts.")
            return Xt

    def preprocess_text(self, text):
        """
        Preprocess a single text by removing punctuation and applying stemming or lemmatization.

        Parameters:
            text (str): The input text.

        Returns:
            str: The preprocessed text.
        """
        pattern = ""
        filtered_words = []
        if self.punctuation or self.stem or self.lemmatize:
            pattern = r"[.,'\"!@#$%^&*(){}?/;`~:\]\[-]"
        text_without_punctuation = re.sub(pattern, "", text)
        stops = set(stopwords.words("english"))
        words = text_without_punctuation.split()
        if self.stem:
            for word in words:
                if word.lower() not in stops:
                    word = self.stemmer.stem(word)
                    filtered_words.append(word)
        elif self.lemmatize:
            for word in words:
                if word.lower() not in stops:
                    word = self.lemmatizer.lemmatize(word)
                    filtered_words.append(word)
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
        # tokenize the text into words
        words = word_tokenize(text)

        # calculate the length of each word
        word_lengths = [len(word) for word in words]

        # calculate the mean word length
        if word_lengths:
            mean_word_length = sum(word_lengths) / len(word_lengths)
        else:
            mean_word_length = 0
        return mean_word_length


class TextWordCounter(BaseEstimator, TransformerMixin):
    """
    Counts words in text data and extracts either frequency distribution features or TF-IDF features.

    Attributes:
        dict_vectorizer (DictVectorizer): The DictVectorizer object.
        tfidf_vectorizer (TfidfVectorizer): The TfidfVectorizer object.

    Parameters:
        freqDict (bool): Whether to extract frequency distribution features.
        bigrams (bool): Whether to include bigrams in the frequency distribution features.

    Methods:
        fit: Fit the TextWordCounter.
        transform: Transform the input text by extracting either frequency distribution features or TF-IDF features.
        count_words: Count words in a text.
    """

    def __init__(self, freqDict=False, bigrams=False):
        self.freqDict = freqDict
        self.bigrams = bigrams
        self.dict_vectorizer = DictVectorizer()
        self.tfidf_vectorizer = TfidfVectorizer()

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
        return self

    def transform(self, X):
        logger.info("Extracting vocabulary size features..")
        vocab_sizes = [self.calculate_vocab_size(text) for text in X]
        vocab_sizes_features = np.array(vocab_sizes).reshape(-1, 1)
        logger.info(f"Shape of vocabulary size features: {vocab_sizes_features.shape}")
        return vocab_sizes_features

    def calculate_vocab_size(self, text):
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
    ):

        self.stopWords = stopWords
        self.errorDetector = errorDetector
        self.punctuationFrequency = punctuationFrequency
        self.sentenceLength = sentenceLength
        self.namedEntity = namedEntity
        self.sentimentAnalysis = sentimentAnalysis
        self.feature_union = []
        self.combined_transformers = None

        # Add transformers based on selected options
        if self.stopWords:
            self.feature_union.append(("stopWords", StopWords()))
        if self.errorDetector:
            self.feature_union.append(("errorDetector", ErrorDetector()))
        if self.punctuationFrequency:
            self.feature_union.append(("punctuationFrequency", PunctuationFrequency()))
        if self.sentenceLength:
            self.feature_union.append(("sentenceLength", SentenceLength()))
        if self.namedEntity:
            self.feature_union.append(("namedEntity", NamedEntity()))
        if self.sentimentAnalysis:
            self.feature_union.append(("sentimentAnalysis", SentimentAnalysis()))
        if self.feature_union:
            self.combined_transformers = FeatureUnion(self.feature_union)

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
            self.feature_union.append(("wordLength", WordLength()))

        if self.vocabularySize:
            self.feature_union.append(("vocabularySize", VocabularySize()))

        self.combined_transformers = FeatureUnion(self.feature_union)

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

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        return result

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
