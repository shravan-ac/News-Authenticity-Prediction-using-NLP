def preprocess(text):
    import re
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    import nltk
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = nltk.word_tokenize(text)
    words = [stemmer.stem(w) for w in words if w not in stop_words]
    return " ".join(words)