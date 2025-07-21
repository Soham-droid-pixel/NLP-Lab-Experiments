import nltk
import spacy
import sklearn_crfsuite
from flair.models import SequenceTagger
from flair.data import Sentence
from nltk.corpus import treebank, conll2000
from nltk.tag import hmm
from nltk.tag import UnigramTagger, BigramTagger, PerceptronTagger
import torch
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# --------------------------- Download Resources ---------------------------
nltk.download('treebank')
nltk.download('averaged_perceptron_tagger')
nltk.download('conll2000')
nltk.download('universal_tagset')

# --------------------------- Load Better Dataset ---------------------------
# Use larger dataset for better training
data = list(treebank.tagged_sents())
# Add CoNLL-2000 data for more diversity
conll_data = list(conll2000.tagged_sents())

# Combine datasets
all_data = data + conll_data
print(f"Total sentences: {len(all_data)}")

# Better train/test split
train_size = int(0.8 * len(all_data))
train_data = all_data[:train_size]
test_data = all_data[train_size:]

# --------------------------- 1. Rule-Based Tagger (spaCy) ---------------------------
try:
    nlp = spacy.load("en_core_web_sm")
    spacy_available = True
except OSError:
    print("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
    spacy_available = False

def spacy_tagger(sent):
    if not spacy_available:
        return ['NN'] * len(sent)
    doc = nlp(" ".join(sent))
    return [token.tag_ for token in doc]

def spacy_accuracy():
    if not spacy_available:
        return 0.0
    correct, total = 0, 0
    for sent in test_data:
        words, tags = zip(*sent)
        pred_tags = spacy_tagger(words)
        # Handle length mismatch
        min_len = min(len(pred_tags), len(tags))
        correct += sum(p == t for p, t in zip(pred_tags[:min_len], tags[:min_len]))
        total += min_len
    return correct / total

# --------------------------- 2. Enhanced CRF Tagger ---------------------------
crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    max_iterations=200,
    c1=0.1,
    c2=0.1
)

def word2features(sent, i):
    word = sent[i][0]
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'BOS': i == 0,
        'EOS': i == len(sent) - 1,
        'word.length': len(word),
    }
    
    # Previous word features
    if i > 0:
        word1 = sent[i-1][0]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
        })
    
    # Next word features
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
        })
    
    return features

print("Training CRF model...")
X_train = [[word2features(s, i) for i in range(len(s))] for s in train_data]
y_train = [[tag for _, tag in s] for s in train_data]
crf.fit(X_train, y_train)

def crf_accuracy():
    X_test = [[word2features(s, i) for i in range(len(s))] for s in test_data]
    y_test = [[tag for _, tag in s] for s in test_data]
    y_pred = crf.predict(X_test)

    correct, total = 0, 0
    for true, pred in zip(y_test, y_pred):
        correct += sum(t == p for t, p in zip(true, pred))
        total += len(true)
    return correct / total

# --------------------------- 3. HMM Tagger ---------------------------
print("Training HMM model...")
trainer = hmm.HiddenMarkovModelTrainer()
hmm_tagger = trainer.train(train_data)

def hmm_accuracy():
    return hmm_tagger.accuracy(test_data)

# --------------------------- 4. Enhanced Flair/BERT Tagger ---------------------------
print("Loading Flair model...")
try:
    # Try simpler model names that are more likely to work
    model_names = [
        "flair/pos-english",
        "flair/pos-english-fast", 
        "pos-english",
        "pos-english-fast",
        "upos-english"
    ]
    
    flair_tagger = None
    for model_name in model_names:
        try:
            print(f"Trying to load: {model_name}")
            flair_tagger = SequenceTagger.load(model_name)
            print(f"Successfully loaded: {model_name}")
            break
        except Exception as e:
            print(f"Failed to load {model_name}: {str(e)[:100]}...")
            continue
    
    if flair_tagger is None:
        raise Exception("No Flair model could be loaded")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    flair_tagger.to(device)
    flair_available = True
    print(f"Flair model loaded on: {device}")
    
except Exception as e:
    print(f"Flair model loading failed: {e}")
    flair_available = False

def flair_accuracy():
    if not flair_available:
        return 0.0
        
    correct, total = 0, 0
    batch_size = 16  # Smaller batch size for stability
    
    # Process fewer sentences to avoid timeout
    test_subset = test_data[:200]
    
    for i, sent in enumerate(test_subset):
        if i % 50 == 0:
            print(f"Processing {i}/{len(test_subset)}...")
            
        words, tags = zip(*sent)
        sentence = Sentence(" ".join(words))
        
        try:
            # Predict with Flair
            flair_tagger.predict(sentence)
            
            # Extract predictions correctly
            pred_tags = []
            for token in sentence:
                # Try different ways to access POS tags
                if hasattr(token, 'get_labels'):
                    labels = token.get_labels()
                    if labels:
                        pred_tags.append(labels[0].value)
                    else:
                        pred_tags.append('NN')
                elif hasattr(token, 'labels') and len(token.labels) > 0:
                    pred_tags.append(token.labels[0].value)
                else:
                    pred_tags.append('NN')
            
            # Handle length mismatch
            min_len = min(len(pred_tags), len(tags))
            if min_len > 0:
                correct += sum(p == t for p, t in zip(pred_tags[:min_len], tags[:min_len]))
                total += min_len
                
        except Exception as e:
            print(f"Error processing sentence {i}: {e}")
            continue
    
    return correct / total if total > 0 else 0.0

# --------------------------- 5. Enhanced NLTK POS Tagger ---------------------------
print("Training NLTK Perceptron model...")
perceptron_tagger = PerceptronTagger(load=False)
perceptron_tagger.train(train_data)

def nltk_accuracy():
    return perceptron_tagger.accuracy(test_data)

# --------------------------- Run & Print Results ---------------------------
print("\n" + "=" * 60)
print("EVALUATING MODELS...")
print("=" * 60)

try:
    spacy_acc = spacy_accuracy()
    print(f"✓ Rule-Based (spaCy) completed: {spacy_acc:.4f}")
except Exception as e:
    spacy_acc = 0.0
    print(f"✗ spaCy failed: {e}")

try:
    crf_acc = crf_accuracy()
    print(f"✓ CRF Tagger completed: {crf_acc:.4f}")
except Exception as e:
    crf_acc = 0.0
    print(f"✗ CRF failed: {e}")

try:
    hmm_acc = hmm_accuracy()
    print(f"✓ HMM Tagger completed: {hmm_acc:.4f}")
except Exception as e:
    hmm_acc = 0.0
    print(f"✗ HMM failed: {e}")

try:
    flair_acc = flair_accuracy()
    print(f"✓ BERT/Flair Tagger completed: {flair_acc:.4f}")
except Exception as e:
    flair_acc = 0.0
    print(f"✗ Flair failed: {e}")

try:
    nltk_acc = nltk_accuracy()
    print(f"✓ NLTK Perceptron completed: {nltk_acc:.4f}")
except Exception as e:
    nltk_acc = 0.0
    print(f"✗ NLTK failed: {e}")

print("\n" + "=" * 60)
print("FINAL RESULTS:")
print("=" * 60)
print(f"1. Rule-Based (spaCy)          Accuracy: {spacy_acc:.4f}")
print(f"2. Enhanced CRF Tagger         Accuracy: {crf_acc:.4f}")
print(f"3. HMM Tagger                  Accuracy: {hmm_acc:.4f}")
print(f"4. BERT/BiLSTM (Flair)         Accuracy: {flair_acc:.4f}")
print(f"5. NLTK Perceptron Tagger      Accuracy: {nltk_acc:.4f}")
print("=" * 60)

# Find best model
models = [
    ("spaCy", spacy_acc),
    ("CRF", crf_acc), 
    ("HMM", hmm_acc),
    ("BERT/Flair", flair_acc),
    ("NLTK Perceptron", nltk_acc)
]
best_model = max(models, key=lambda x: x[1])
print(f"🏆 Best Model: {best_model[0]} with {best_model[1]:.4f} accuracy")
