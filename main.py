import os
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def clean_chat_text(text):
    text = re.sub(r'^(User:|AI:)\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    filler_words = set([
        'whats', 'good', 'like', 'favorite', 'think', 'also', 'yes', 'no', 'hey', 'oh',
        'you', 'i', 'me', 'we', 'us', 'im', 'its', 'dont', 'would', 'could', 'maybe'
    ])
    tokens = text.split()
    filtered_tokens = [w for w in tokens if w not in filler_words]
    return ' '.join(filtered_tokens)

def extract_keywords_tfidf(text, top_n=10):
    stop_words = list(stopwords.words('english'))
    vectorizer = TfidfVectorizer(stop_words=stop_words, ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.toarray()[0]
    sorted_indices = tfidf_scores.argsort()[-top_n:][::-1]
    top_keywords = [feature_names[i] for i in sorted_indices]
    weak_words = {'think', 'makes', 'like', 'know', 'would', 'could', 'also', 'yes', 'no', 'hey', 'oh'}
    filtered_keywords = [w for w in top_keywords if w not in weak_words]
    return filtered_keywords

def get_topics_from_text(text, num_topics=2, top_n_words=5):
    stop_words = list(stopwords.words('english'))
    vectorizer = CountVectorizer(stop_words=stop_words, max_features=1000)
    dtm = vectorizer.fit_transform([text])
    lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda_model.fit(dtm)
    feature_names = vectorizer.get_feature_names_out()
    topic_scores = lda_model.components_.sum(axis=1)
    main_topic_index = topic_scores.argmax()
    topic = lda_model.components_[main_topic_index]
    top_indices = topic.argsort()[-top_n_words:][::-1]
    topic_words = [feature_names[i] for i in top_indices]
    return topic_words

def count_exchanges(chat_content):
    lines = [line.strip() for line in chat_content.split('\n') if line.strip()]
    exchange_count = sum(1 for line in lines if line.startswith("User:") or line.startswith("AI:"))
    return exchange_count

def main():
    chat_file_path = 'sample_chats/chat.txt'
    output_file_path = 'output/summary.txt'

    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    try:
        with open(chat_file_path, 'r', encoding='utf-8') as f:
            chat_content = f.read()
    except FileNotFoundError:
        print(f"Error: Chat file not found at {chat_file_path}")
        return

    total_exchanges = count_exchanges(chat_content)
    cleaned_text = clean_chat_text(chat_content)
    top_keywords = extract_keywords_tfidf(cleaned_text, top_n=10)
    main_topic_keywords = get_topics_from_text(cleaned_text, num_topics=1, top_n_words=5)

    summary_output = f"Summary:\n"
    summary_output += f"- The conversation had {total_exchanges} exchanges.\n"

    if main_topic_keywords:
        summary_output += f"- The conversation was mainly about {', '.join(main_topic_keywords)}.\n"
    else:
        summary_output += "- The main topic could not be clearly determined.\n"

    summary_output += f"- Most common keywords: {', '.join(top_keywords[:5])}\n"

    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write(summary_output)
        print(f"Analysis complete. Summary saved to {output_file_path}")
    except IOError:
        print(f"Error: Could not write to output file {output_file_path}")

if __name__ == "__main__":
    main()
