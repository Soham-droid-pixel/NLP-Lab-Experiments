import re
import string
import nltk
import spacy

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, LancasterStemmer, WordNetLemmatizer

# Load tools
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
spacy_model = spacy.load("en_core_web_sm")

# Preprocessing tools
stopwords_nltk = set(stopwords.words("english"))
stopwords_spacy = spacy_model.Defaults.stop_words
porter = PorterStemmer()
lancaster = LancasterStemmer()
lemmatizer = WordNetLemmatizer()

# Get input sentences
sentences = []
for i in range(10):
    sentence = input(f"Enter sentence {i+1}: ")
    sentences.append(sentence)

# Process each sentence
for i, sentence in enumerate(sentences):
    print(f"\n--- Sentence {i+1} ---")
    print("Original:", sentence)

    # Step 1: Clean text
    cleaned = re.sub(r'\d+', '', sentence.lower().translate(str.maketrans('', '', string.punctuation)))
    print("Cleaned:", cleaned)

    # Step 2: Tokenization
    words = word_tokenize(cleaned)
    print("Word Tokens:", words)

    # Step 3: Subword and Character tokens
    subwords = [cleaned[i:i+3] for i in range(len(cleaned)-2)]
    chars = list(cleaned.replace(" ", ""))
    print("Subword Tokens (3-gram):", subwords)
    print("Character Tokens:", chars)

    # Step 4: Stopword removal
    nltk_no_stop = [w for w in words if w not in stopwords_nltk]
    spacy_no_stop = [w for w in words if w not in stopwords_spacy]
    print("Without Stopwords (NLTK):", nltk_no_stop)
    print("Without Stopwords (SpaCy):", spacy_no_stop)

    # Step 5: Stemming
    porter_stem = [porter.stem(w) for w in nltk_no_stop]
    lancaster_stem = [lancaster.stem(w) for w in nltk_no_stop]
    print("Porter Stemmed:", porter_stem)
    print("Lancaster Stemmed:", lancaster_stem)

    # Step 6: Lemmatization
    lemmatized = [lemmatizer.lemmatize(w) for w in nltk_no_stop]
    print("Lemmatized:", lemmatized)
    
    # ------------------ NLP Preprocessing Pipeline Explanation ------------------

# ✔ Input:
#   - Takes 10 user-defined sentences via input()

# ✔ Step 1: Noise Removal (Cleaning)
#   - Converts sentence to lowercase
#   - Removes digits using regex
#   - Removes punctuation using string.punctuation

# Example:
#   "The quick brown fox jumps over 2 lazy dogs!"
#   → "the quick brown fox jumps over  lazy dogs"

# ✔ Step 2: Tokenization
#   - Word Tokenization → word_tokenize()
#     Output: ['the', 'quick', 'brown', 'fox', 'jumps', ...]
#   - Subword Tokenization → 3-character substrings (sliding window)
#     Output: ['the', 'he ', 'e q', ' qu', 'qui', ...]
#   - Character Tokenization → list of all characters (excluding spaces)
#     Output: ['t', 'h', 'e', 'q', 'u', 'i', 'c', 'k', ...]

# ✔ Step 3: Stopword Removal
#   - Removes common words (e.g., "the", "is", "and")
#   - Uses both:
#     1. NLTK stopword list
#     2. SpaCy built-in stopwords

# ✔ Step 4: Stemming (reduces words to base/root)
#   - Porter Stemmer → Mild (e.g., "jumps" → "jump", "lazy" → "lazi")
#   - Lancaster Stemmer → Aggressive (e.g., "lazy" → "lazy", but more cutting)

# ✔ Step 5: Lemmatization (returns actual dictionary root words)
#   - Uses NLTK WordNetLemmatizer
#   - Example: "dogs" → "dog", "jumps" → "jump"

# ✔ Output:
#   - Shows each transformation stage for every sentence
#   - Helps in comparison & analysis of preprocessing steps

# ---------------------------------------------------------------------------

