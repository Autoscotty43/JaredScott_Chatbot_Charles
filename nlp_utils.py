import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

def process_text(text):
    text = text.lower()
    words = word_tokenize(text)
    words = [word for word in words if word.isalnum()]
    words = [word for word in words if word not in stopwords.words("english")]
    return " ".join(words)

def clean_response(response):
    unwanted_phrases = [
        "I am a student at a university",
        "As a student",
        "I study at a university",
        "I'm an AI language model",
        "I'm an AI and don't have personal experiences.",
        "I am an AI language model",
        "It's difficult for me to answer",
        "I cannot answer",
        "charles:",
        "</s>", # Remove end of sequence token
        "<pad>"
    ]
    for phrase in unwanted_phrases:
        response = response.replace(phrase, "")
    return response.strip()