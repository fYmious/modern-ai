import random
import sys

import nltk

nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download("brown")
nltk.download("names")
nltk.download("movie_reviews")
nltk.download("stopwords")

TEXT_INPUT: str = ""
with open("data/data.txt", 'r') as input_file:
    TEXT_INPUT = "".join(input_file.readlines())

def task1() -> None:
    from nltk.tokenize import sent_tokenize, word_tokenize, WordPunctTokenizer

    sentences: list[str] = sent_tokenize(TEXT_INPUT);
    print(sentences)
    print(len(sentences))

    words: list[str] = word_tokenize(TEXT_INPUT)
    print(words)
    print(len(words))


    wpt: WordPunctTokenizer = WordPunctTokenizer()
    tokenized: list[str] = WordPunctTokenizer.tokenize(wpt, TEXT_INPUT)
    print(tokenized)
    print(len(tokenized))

def task2() -> None:
    from nltk.stem.porter import PorterStemmer
    from nltk.stem.lancaster import LancasterStemmer
    from nltk.stem.snowball import SnowballStemmer
    
    roots: dict[str, list[str]] = dict()
    words: list[str] = nltk.word_tokenize(TEXT_INPUT)
    stemmed_words: list[str] = words.copy()

    ps: PorterStemmer = PorterStemmer()

    for i in range(len(words)):
        word: str = words[i]
        stemmed: str = ps.stem(words[i])
        stemmed_words[i] = stemmed
        if stemmed in roots.keys():
            roots[stemmed] += [word]
        else:
            roots[stemmed] = [word]

    print_roots(roots)

    roots = dict()
    ls: LancasterStemmer = LancasterStemmer()

    for i in range(len(words)):
        word: str = words[i]
        stemmed: str = ls.stem(words[i])
        stemmed_words[i] = stemmed
        if stemmed in roots.keys():
            roots[stemmed] += [word]
        else:
            roots[stemmed] = [word]

    print_roots(roots)

    roots = dict()
    ss: SnowballStemmer = SnowballStemmer("english")

    for i in range(len(words)):
        word: str = words[i]
        stemmed: str = ss.stem(words[i])
        stemmed_words[i] = stemmed
        if stemmed in roots.keys():
            roots[stemmed] += [word]
        else:
            roots[stemmed] = [word]

    print_roots(roots)

def print_roots(roots: dict[str, list[str]]) -> None:
    for root in roots.keys():
        same_root: list[str] = roots[root]
        if len(set([w.lower() for w in same_root])) > 1:
            print(root, " ", roots[root])

def task3() -> None:
    from nltk.stem import WordNetLemmatizer

    wnl: WordNetLemmatizer = WordNetLemmatizer()

    roots: dict[str, list[str]] = dict()
    words: list[str] = nltk.word_tokenize(TEXT_INPUT)
    stemmed_words: list[str] = words.copy()

    for i in range(len(words)):
        word: str = words[i]
        stemmed: str = wnl.lemmatize(words[i], pos='n')
        stemmed_words[i] = stemmed
        if stemmed in roots.keys():
            roots[stemmed] += [word]
        else:
            roots[stemmed] = [word]

    print_roots(roots)

    roots = dict()

    for i in range(len(words)):
        word: str = words[i]
        stemmed: str = wnl.lemmatize(words[i], pos='v')
        stemmed_words[i] = stemmed
        if stemmed in roots.keys():
            roots[stemmed] += [word]
        else:
            roots[stemmed] = [word]

    print_roots(roots)
    
def chunk_words(words_list, chunk_size):
  for i in range(0, len(words_list), chunk_size):
    yield words_list[i : i + chunk_size]

def task4(n: int, m: int):
    from nltk.corpus import brown

    words = brown.words()[:n]

    chunks = list(chunk_words(words, m))

    for index, chunk in enumerate(chunks[:20]):
      chunk_text = " ".join(chunk)

      if len(chunk_text) > 50:
        display_text = chunk_text[:50] + "..."
      else:
        display_text = chunk_text

      print(f"{index + 1}: {display_text}")

    return chunks

def task5(n: int = 1000, m: int = 200) -> None:
    from sklearn.feature_extraction.text import CountVectorizer
    from nltk.corpus import brown
    import numpy as np

    words = brown.words()[:n]
    chunks = [
        {"index": i, "text": " ".join(chunk)}
        for i, chunk in enumerate(chunk_words(words, m))
    ]
    texts: list[str] = [chunk["text"] for chunk in chunks]

    vectorizer: CountVectorizer = CountVectorizer()
    doc_term_matrix = vectorizer.fit_transform(texts)
    vocabulary: list[str] = list(vectorizer.get_feature_names_out())

    print("# of chunks:", len(chunks))
    print("dict size:", len(vocabulary))
    print("doc term matrix shape:", doc_term_matrix.shape)
    print()

    dense = doc_term_matrix.toarray()
    print(dense)
    print()

    totals = dense.sum(axis=0)
    top_ids = np.argsort(-totals)[:10]

    header = "chunk".ljust(8) + "".join(vocabulary[j].ljust(8) for j in top_ids)
    print(header)
    for i in range(dense.shape[0]):
        row = str(i).ljust(8) + "".join(str(dense[i][j]).ljust(8) for j in top_ids)
        print(row)

def task6() -> None:
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.feature_extraction.text import CountVectorizer

    # Ключи категорий - имена из пакета scikit-learn
    category_map: dict[str, str] = {
        "rec.sport.hockey": "Hockey",
        "sci.space": "Space",
        "comp.graphics": "Graphics",
        "talk.politics.guns": "Politics (guns)",
        "sci.med": "Medicine",
    }
    print("categories:", category_map)

    training_data = fetch_20newsgroups(
        subset="train",
        categories=category_map.keys(),
        shuffle=True,
        random_state=7,
    )
    print("training data size:", len(training_data.data))

    count_vectorizer: CountVectorizer = CountVectorizer()
    train_tc = count_vectorizer.fit_transform(training_data.data)
    print("train tc shape:", train_tc.shape)

    tfidf: TfidfTransformer = TfidfTransformer()
    train_tfidf = tfidf.fit_transform(train_tc)

    input_data: list[str] = [
        "The goalie made an amazing save and the team scored in overtime",
        "NASA is planning to launch a new rocket to the Moon next year",
        "I need a better video card to render 3D graphics and textures",
        "The doctor prescribed antibiotics to treat the infection",
        "The government wants to restrict the right to own a gun",
    ]

    classifier: MultinomialNB = MultinomialNB().fit(train_tfidf, training_data.target)

    input_tc = count_vectorizer.transform(input_data)
    input_tfidf = tfidf.transform(input_tc)

    predictions = classifier.predict(input_tfidf)

    for sent, category in zip(input_data, predictions):
        print("\nInput:", sent)
        print("Predicted category:", category_map[training_data.target_names[category]])

def last_letters(word: str, num_letters: int) -> dict[str, str]:
    return {"feature": word[-num_letters:].lower()}

def task7() -> None:
    from nltk import NaiveBayesClassifier
    from nltk.classify import accuracy as nltk_accuracy
    from nltk.corpus import names

    male_list = [(name, "male") for name in names.words("male.txt")]
    female_list = [(name, "female") for name in names.words("female.txt")]
    data = male_list + female_list

    random.seed(7)
    random.shuffle(data)

    test_names: list[str] = [
        "Anna", "Maria", "Olga", "Elena",
        "Nikita", "Ivan", "Dmitry", "Mark",
    ]

    train_size: int = int(0.8 * len(data))
    train_names, test_set = data[:train_size], data[train_size:]
    print("Обучающая выборка:", len(train_names), "Тестовая выборка:", len(test_set))

    best_n: int = 1
    best_accuracy: float = 0.0

    for n in range(1, 6):
        train_features = [(last_letters(name, n), gender) for name, gender in train_names]
        test_features = [(last_letters(name, n), gender) for name, gender in test_set]

        classifier = NaiveBayesClassifier.train(train_features)
        acc: float = nltk_accuracy(classifier, test_features)
        print(f"\nN = {n}, accuracy = {round(100 * acc, 2)}%")

        if acc > best_accuracy:
            best_accuracy = acc
            best_n = n

        for name in test_names:
            print(f"  {name}: {classifier.classify(last_letters(name, n))}")

    print(f"\n optimal N = {best_n} (accuracy = {round(100 * best_accuracy, 2)}%)")

def extract_features(words: list[str]) -> dict[str, bool]:
    return dict([(word, True) for word in words])

def task8() -> None:
    from nltk.corpus import movie_reviews
    from nltk.classify import NaiveBayesClassifier
    from nltk.classify.util import accuracy as nltk_accuracy

    fileids_pos = movie_reviews.fileids("pos")
    fileids_neg = movie_reviews.fileids("neg")

    features_pos = [
        (extract_features(movie_reviews.words(fileids=[f])), "Positive")
        for f in fileids_pos
    ]
    features_neg = [
        (extract_features(movie_reviews.words(fileids=[f])), "Negative")
        for f in fileids_neg
    ]

    num_pos: int = int(0.8 * len(features_pos))
    num_neg: int = int(0.8 * len(features_neg))

    features_train = features_pos[:num_pos] + features_neg[:num_neg]
    features_test = features_pos[num_pos:] + features_neg[num_neg:]

    print("features train:", len(features_train))
    print("features test:", len(features_test))

    classifier = NaiveBayesClassifier.train(features_train)
    print("accuracy (train):", round(100 * nltk_accuracy(classifier, features_train), 2), "%")
    print("accuracy (test):", round(100 * nltk_accuracy(classifier, features_test), 2), "%")

    print("\n20 most informative words:")
    classifier.show_most_informative_features(20)

    input_reviews: list[str] = [
        "The costumes in this movie were great",
        "I think the story was terrible and the characters were very weak",
        "People say that the director of the movie is amazing",
        "This is such an idiotic movie. I will not recommend it to anyone",
        "The acting was wonderful and the plot kept me interested until the very end",
    ]

    print("\npredictions:")
    for review in input_reviews:
        probdist = classifier.prob_classify(extract_features(nltk.word_tokenize(review.lower())))
        pred_sentiment = probdist.max()
        print("\nReview:", review)
        print("Predicted sentiment:", pred_sentiment)
        print("Probability:", round(probdist.prob(pred_sentiment), 4))
        print("  Positive:", round(probdist.prob("Positive"), 4),
              "| Negative:", round(probdist.prob("Negative"), 4))

def task9() -> None:
    from nltk.tokenize import RegexpTokenizer
    from nltk.corpus import stopwords
    from nltk.stem.snowball import SnowballStemmer
    from gensim import models, corpora

    tokenizer: RegexpTokenizer = RegexpTokenizer(r"\w+")
    stop_words: set[str] = set(stopwords.words("english"))
    stemmer: SnowballStemmer = SnowballStemmer("english")

    tokens: list[list[str]] = []
    for line in TEXT_INPUT.split("\n"):
        if not line.strip():
            continue
        line_tokens = tokenizer.tokenize(line.lower())
        line_tokens = [t for t in line_tokens if t not in stop_words]
        line_tokens = [stemmer.stem(t) for t in line_tokens]
        tokens.append(line_tokens)

    dict_tokens = corpora.Dictionary(tokens)
    doc_term_mat = [dict_tokens.doc2bow(token) for token in tokens]

    num_topics: int = 2
    ldamodel = models.ldamodel.LdaModel(
        doc_term_mat,
        num_topics=num_topics,
        id2word=dict_tokens,
        passes=25,
        random_state=42,
    )

    print("topics & 5 representative words:")
    for topic_id in range(num_topics):
        print(f"\nТема {topic_id}:")
        for word, weight in ldamodel.show_topic(topic_id, topn=5):
            print(f"  {word}: {round(float(weight), 4)}")

    print("\n topic destribution by sentences:")
    for i, bow in enumerate(doc_term_mat):
        print(i + 1, [(t, round(float(p), 3)) for t, p in ldamodel.get_document_topics(bow)])

def task10() -> None:
    import numpy as np
    import torch
    from datasets import load_dataset
    from sklearn.metrics import accuracy_score, f1_score
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        pipeline,
    )

    MODEL_NAME: str = "distilbert-base-uncased"
    SEED: int = 42

    # dataset = load_dataset("imdb")
    dataset = load_dataset("stanfordnlp/imdb")
    train_full = dataset["train"].shuffle(seed=SEED).select(range(4000))
    test_ds = dataset["test"].shuffle(seed=SEED).select(range(1000))

    split = train_full.train_test_split(test_size=0.1, seed=SEED)
    train_ds, val_ds = split["train"], split["test"]

    print(dataset)
    print("train:", len(train_ds), "val:", len(val_ds), "test:", len(test_ds))
    print("example:", train_ds[0]["text"][:200], "| label =", train_ds[0]["label"])

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=256)

    train_ds = train_ds.map(tokenize, batched=True)
    val_ds = val_ds.map(tokenize, batched=True)
    test_ds = test_ds.map(tokenize, batched=True)

    id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    label2id = {"NEGATIVE": 0, "POSITIVE": 1}
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, id2label=id2label, label2id=label2id
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds),
        }

    args = TrainingArguments(
        output_dir="results",
        num_train_epochs=2,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        seed=SEED,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    val_metrics = trainer.evaluate(val_ds)
    print("\nvalidation metrics:")
    print("  accuracy:", round(val_metrics["eval_accuracy"], 4))
    print("  f1:", round(val_metrics["eval_f1"], 4))

    test_metrics = trainer.evaluate(test_ds)
    print("\ntest metrics:")
    print("  accuracy:", round(test_metrics["eval_accuracy"], 4))
    print("  f1:", round(test_metrics["eval_f1"], 4))

    clf = pipeline(
        "text-classification",
        model=trainer.model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
    )

    custom_texts: list[str] = [
        "This movie was absolutely fantastic, I loved every minute of it!",
        "A complete waste of time. The plot made no sense and the acting was awful.",
        "The film is fine, nothing special, but not bad either.",
        "I expected much more from this director, what a disappointment.",
        "Brilliant performances and a touching story, highly recommended.",
        "Not good. Not bad. Just boring.",
    ]

    print("\ncustom text predictions:")
    for text, result in zip(custom_texts, clf(custom_texts)):
        print(f"\nText: {text}")
        print(f"Label: {result['label']}, score: {round(result['score'], 4)}")

TASKS = {
    "1": task1,
    "2": task2,
    "3": task3,
    "4": lambda: task4(1000, 15),
    "5": task5,
    "6": task6,
    "7": task7,
    "8": task8,
    "9": task9,
    "10": task10,
}

if __name__ == "__main__":
    to_run: list[str] = sys.argv[1:] if len(sys.argv) > 1 else list(TASKS.keys())
    for number in to_run:
        TASKS[number]()
