import pandas as pd
from sklearn.model_selection import train_test_split
import os
import re
import contractions
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from scripts.utils import load_config, get_logger, load_params


logger = get_logger("baseline", "baseline.log")


def clean_text(text: str) -> str:
    """Clean and preprocess a text string."""
    text = str(text).lower()
    text = re.sub(r'<.*?>','',text)
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'[^A-Za-z0-9\s]','',text)
    text = contractions.fix(text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Simple word tokenization
    tokens = text.split()
    
    STOPWORDS = set(stopwords.words("english"))
    tokens = [token for token in tokens if token not in STOPWORDS]
    lemma = WordNetLemmatizer()
    tokens = [lemma.lemmatize(token) for token in tokens]
    text = " ".join(tokens)
    return text

def plot_eda(df: pd.DataFrame, text_col: str, sentiment_col: str, out_dir: str, sentiment_labels: dict) -> None:
    os.makedirs(out_dir, exist_ok=True)
    # Data quality
    plt.figure(figsize=(6,4))
    df.isnull().sum().plot(kind='bar')
    plt.title('Missing Values per Column')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'missing_values.png'))
    plt.close()

    # Class imbalance
    plt.figure(figsize=(6,4))
    sentiment_counts = df[sentiment_col].value_counts()
    plt.bar(sentiment_counts.index.map(sentiment_labels), sentiment_counts.values)
    plt.title('Class Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'class_distribution.png'))
    plt.close()

    # Most common words
    all_words = ' '.join(df[text_col]).split()
    word_freq = pd.Series(all_words).value_counts().head(20)
    plt.figure(figsize=(8,4))
    word_freq.plot(kind='bar')
    plt.title('Top 20 Most Common Words')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'common_words.png'))
    plt.close()

    # Word cloud
    wc = WordCloud(width=800, height=400, background_color='white').generate(' '.join(all_words))
    plt.figure(figsize=(10,5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'wordcloud.png'))
    plt.close()

def preprocess_data(config: dict) -> None:
    """
    Reads raw data, performs basic cleaning, splits it into train and test sets,
    and saves them to the processed data folder.
    """
    data_path = config['data']['raw']
    processed_data_path = config['data']['processed']
    text_col = config['data']['columns']['text']
    sentiment_col = config['data']['columns']['sentiment']
    eda_dir = config['data']['eda_dir']
    
    # Load data and parameters
    df = pd.read_csv(data_path)
    params = load_params()
    sentiment_labels = params.get('sentiment_labels', {0: "Negative", 1: "Neutral", 2: "Positive"})
    valid_sentiments = list(sentiment_labels.keys())
    baseline_params = params['baseline']
    
    # Load configuration
    config = load_config()
    nltk_data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../', config['data']['nltk_data']))
    if os.path.exists(nltk_data_path):
        # Insert at the beginning of the search path to prioritize our local data
        nltk.data.path.insert(0, nltk_data_path)

    # Download required NLTK data if not already present
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)

    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)
    
    # Clean text and validate data
    df[text_col] = df[text_col].astype(str).apply(clean_text)
    
    # Drop rows with missing values or invalid sentiment labels
    initial_size = len(df)
    df = df.dropna(subset=[text_col, sentiment_col])
    df = df[df[sentiment_col].isin(valid_sentiments)]
    df = df[df[text_col] != '']
    
    # Log data cleaning statistics
    logger.info(f"Initial dataset size: {initial_size}")
    logger.info(f"After cleaning: {len(df)}")
    logger.info(f"Removed {initial_size - len(df)} rows")
    logger.info("\nClass distribution after cleaning:")
    for sentiment_id, label in sentiment_labels.items():
        count = len(df[df[sentiment_col] == sentiment_id])
        logger.info(f"{label}: {count} ({count/len(df)*100:.2f}%)")

    # EDA
    os.makedirs(eda_dir, exist_ok=True)
    plot_eda(df, text_col, sentiment_col, eda_dir, sentiment_labels)

    # Split data
    test_size = params.get('test_size', 0.2)
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42, stratify=df[sentiment_col])

    # Save processed data
    os.makedirs(processed_data_path, exist_ok=True)
    train_df.to_csv(os.path.join(processed_data_path, 'train.csv'), index=False)
    test_df.to_csv(os.path.join(processed_data_path, 'test.csv'), index=False)

    logger.info(f"Experiment: {baseline_params.get('model', 'unknown')}")
    logger.info(f"Number of training samples: {len(train_df)}")
    logger.info(f"Number of test samples: {len(test_df)}")

if __name__ == '__main__':
    config = load_config()
    preprocess_data(config)
