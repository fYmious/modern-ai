#show link: underline

#set document(
  title: [ЛР2],
)

#set heading(numbering: "1.")

#let img(imagePath, caption, supplement: [Рисунок]) = {
  align(center)[
    #figure(image(imagePath), caption: caption, supplement: supplement)
  ]
}

#let ph(t) = highlight(fill: yellow)[#t]

#show figure.where(kind: table): set figure.caption(position: top)

#let tbl(caption, ..args) = figure(
  caption: caption,
  kind: table,
  supplement: [Таблица],
  {
    set par(first-line-indent: 0pt, justify: false)
    table(..args)
  },
)


#set page(
  paper: "a4",
  numbering: "1",
)

#set par(
  justify: true,
  first-line-indent: (
    amount: 1.25cm,
    all: true,
  ),
  spacing: 0.65em,
)

#set text(
  lang: "ru",
  font: "Times New Roman",
  size: 14pt,
)

#set page(footer: context {
  if counter(page).get().first() > 1 [
    #align(center)[
      #counter(page).display("1")
    ]
  ]
  if counter(page).get().first() == 1 [
    #align(center)[
      Санкт-Петербург \ 2026
    ]
  ]
})

#set page(header: context {
  if counter(page).get().first() == 1 [
    #align(center)[
      *Министерство науки и высшего образования Российской Федерации* \
    ]
  ]
})

#show raw: set text(font: "Consolas")
#show raw.where(block: false): box.with(
  fill: luma(240),
  inset: (x: 3pt, y: 0pt),
  outset: (y: 3pt),
  radius: 2pt,
)

#show raw.where(block: true): block.with(
  fill: luma(240),
  inset: 10pt,
  radius: 4pt,
)

// title

#align(center)[
  ФЕДЕРАЛЬНОЕ ГОСУДАРСТВЕННОЕ АВТОНОМНОЕ ОБРАЗОВАТЕЛЬНОЕ УЧРЕЖДЕНИЕ ВЫСШЕГО ОБРАЗОВАНИЯ
]

#align(center)[
  #text(size: 12pt)[
    \ *НАЦИОНАЛЬНЫЙ ИССЛЕДОВАТЕЛЬСКИЙ УНИВЕРСИТЕТ ИТМО*
  ]
]

#align(center)[
  \ *ITMO University*
]


#for _ in range(5) { linebreak() }

#align(center)[*ЛАБОРАТОРНАЯ 2*]

#table(
  stroke: white,
  columns: 1,
  inset: 10pt,

  [*По дисциплине* Современные методы искусственного интеллекта ],
  [*Тема работы* Решение задач NLP с помощью библиотеки Hugging Face],
  [*Обучающийся* Дощенников Никита Андреевич],
  [*Факультет* Прикладной информатики],
  [*Группа* К3321],
  [*Направление подготовки* 11.03.02 Инфокоммуникационные технологии и системы связи],
  [*Образовательная программа* Программирование в инфокоммуникационных системах],
  [],
)

#table(
  stroke: white,
  columns: 4,
  inset: 10pt,

  table.cell(align: top)[*Обучающийся*],

  table(
    columns: 1,
    inset: 2pt,
    stroke: white,
    table.cell(
      align: top + center,
    )[#text(size: 10pt, fill: white)[.]],
    [#line(length: 100pt)],
  ),

  table(
    columns: 1,
    inset: 2pt,
    stroke: white,
    table.cell(
      align: top + center,
    )[#text(size: 11pt, fill: white)[.]],
    [#line(length: 100pt)],
  ),

  table(
    columns: 1,
    inset: 2pt,
    stroke: white,
    table.cell(
      align: top + center,
    )[#text(size: 10pt)[Дощенников Н.А.]],
    [#line(length: 100pt)],
  ),

  [],
  table.cell(align: center)[#text(size: 10pt)[(дата)]],
  table.cell(align: center)[#text(size: 10pt)[(подпись)]],
  table.cell(align: center)[#text(size: 10pt)[(Ф.И.О.)]],

  table.cell(align: top)[*Руководитель*],

  table(
    columns: 1,
    inset: 2pt,
    stroke: white,
    table.cell(
      align: top + center,
    )[#text(size: 10pt, fill: white)[.]],
    [#line(length: 100pt)],
  ),

  table(
    columns: 1,
    inset: 2pt,
    stroke: white,
    table.cell(
      align: top + center,
    )[#text(size: 11pt, fill: white)[.]],
    [#line(length: 100pt)],
  ),

  table(
    columns: 1,
    inset: 2pt,
    stroke: white,
    table.cell(
      align: top + center,
    )[#text(size: 10pt)[Царев Д.В.]],
    [#line(length: 100pt)],
  ),

  [],
  table.cell(align: center)[#text(size: 10pt)[(дата)]],
  table.cell(align: center)[#text(size: 10pt)[(подпись)]],
  table.cell(align: center)[#text(size: 10pt)[(Ф.И.О.)]],
)


#pagebreak()

#outline(title: [Содержание])

#pagebreak()
= Цель

Целью данной работы является получение практических навыков работы с современными архитектурами трансформеров для обработки текстовой информации.

#pagebreak()
= Задание 1. Токенизация текстовых данных

Сначала были импортированы следующие пакеты:

```py
from nltk.tokenize import sent_tokenize, word_tokenize, WordPunctTokenizer
```

А также установлены зависимости:

```sh
pip install nltk
```

Затем был инициирован входной текст и записан в переменную `TEXT_INPUT`:

```py
TEXT_INPUT: str = ""
with open("data/data.txt", 'r') as input_file:
    TEXT_INPUT = "".join(input_file.readlines())
```

Текст был разбит на токены с помощью 3-x токенизаторов:

== Токенизатор предложений `sent_tokenize`

```py
sentences = sent_tokenize(TEXT_INPUT);
print(sentences)
```

#img("assets/1.png", [Разбивка по предложениям])

Данные были разбиты на 10 предложений.

== Токенизатор слов `word_tokenize` <wordTokenize>

```py
words = word_tokenize(TEXT_INPUT)
print(words)
```

#img("assets/2.png", [Разбивка по словам])

Данные были разделены по словам и знакам пунктуации. Получилось 177 фрагментов.

== Токенизатор пунктуации и слов `WordPunctTokenizer`

```py
wpt = WordPunctTokenizer()
tokenized = WordPunctTokenizer.tokenize(wpt, TEXT_INPUT)
print(tokenized)
```

#img("assets/3.png", [Разбивка по словам и знакам препинания])

Данные были разбиты на 177 фрагментов, идентичных тем, что получились при разделении по словам в @wordTokenize.

В результате всех 3-x разбиений получились ожидаемые результаты разбивки.

Сравнение 3-x способов токенизации:

- `sent_tokenize` работает на уровне предложений. Tекст был разбит на 10 предложений, по числу строк во входном файле. Разделителем служит точка в конце предложения, при этом токенизатор обучен отличать её от точек в сокращениях;
- `word_tokenize` выделяет отдельные слова и отделяет знаки препинания в самостоятельные токены. Этот токенизатор также учитывает особенности английского языка, например, разбивает сокращения вида `don't` на `do` и `n't`;
- `WordPunctTokenizer` разбивает текст по регулярному выражению, отделяя последовательности букв и цифр от последовательностей знаков препинания. В данном тексте нет апострофов, дефисов и сокращений, поэтому результат полностью совпал с `word_tokenize`. Различия проявились бы, например, на слове `don't`, которое `WordPunctTokenizer` разобьет на три токена: `don`, `'` и `t`.

Таким образом, для задач, где важны границы предложений, следует использовать `sent_tokenize`, а для получения слов предпочтительнее `word_tokenize`, так как он лучше работает с особенностями английской грамматики.

#pagebreak()
= Задание 2. Преобразование слов в их базовые формы с помощью стемминга

Для задания были импортированы следующие пакеты:

```py
from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.snowball import SnowballStemmer
```

Затем слова были приведены к корневой форме с помощью 3-х стеммеров:

== PorterStemmer

```py
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
```

#img("assets/4.png", [Результат стемминга `PorterStemmer`])

Стеммер Портера сгруппировал словоформы по 8 корням. Были объединены формы одного слова (`empire` и `empires`, `time` и `times`, `set` и `sets`), а также однокоренные слова разных частей речи (`expanded` и `expand`, `historical` и `Historically`, `mathematics` и `mathematical`). Получившиеся корни не всегда являются словами (`empir`, `structur`, `cultur`, `histor`), поскольку стеммер только отсекает окончания по набору правил. Слова `Mathematicians` и `history` в свои группы не попали, стеммер оставил для них отличающиеся основы (`mathematician` и `histori`).

== LancasterStemmer

```py
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
```

#img("assets/5.png", [Результат стемминга `LancasterStemmer`])

Стеммер Ланкастера оказался самым агрессивным и сократил слова сильнее остальных (`tim`, `cult`, `hist`, `mathem`). За счет этого в группы попало больше слов, в частности `history` объединилось с `historical` и `Historically`, а `Mathematicians` с `mathematics` и `mathematical`. Однако у такой агрессивности есть обратная сторона, например, в одну группу `on` попали несвязанные по смыслу слова `one`, `on` и `only`, а `Late` и `later` были сведены к `lat`.

== SnowballStemmer

```py
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
```
#img("assets/6.png", [Результат стемминга `SnowballStemmer`])

Результат `SnowballStemmer` на данном тексте полностью совпал с `PorterStemmer`. Получились те же 8 групп и те же корни. Различия между этими двумя алгоритмами проявляются на более широком наборе слов, так как Snowball содержит ряд исправлений правил Портера и поддерживает не только английский, но и другие языки.

Итоги сравнения стеммеров:

- `PorterStemmer` и `SnowballStemmer` дали одинаковый, сбалансированный результат. Ошибочных объединений нет, но часть однокоренных слов осталась неразделенной;
- `LancasterStemmer` сильнее всех сокращает слова и объединяет больше форм, но допускает ошибки;
- во всех случаях получаемые корни не являются словами, поэтому для задач, где нужны осмысленные слова, стемминг не подходит. Преимущество стемминга в простоте и скорости, он не использует словарь.

#pagebreak()
= Задание 3. Преобразование слов в их базовые формы с помощью лемматизации

Были лемматизированны входные слова в двух режимах:

== Существительное `lemmatizer.lemmatize(word, pos='n')`

```py
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
```

#img("assets/7.png", [Результат лемматизации (существительное)])

В режиме существительного лемматизатор привел к единственному числу формы `empires`, `times`, `structures`, `sets`. Глагольные формы остались без изменений, так как они не являются формами существительных. Кроме того, возникла ошибка, а именно, слово `as` было сведено к `a`, поскольку WordNet трактует `as` как множественное число существительного `a`. В отличие от стеммеров, все получившиеся леммы являются настоящими словами.

== Глагол `lemmatizer.lemmatize(word, pos='v')`

```py
for i in range(len(words)):
    word: str = words[i]
    stemmed: str = wnl.lemmatize(words[i], pos='v')
    stemmed_words[i] = stemmed
    if stemmed in roots.keys():
        roots[stemmed] += [word]
    else:
        roots[stemmed] = [word]

print_roots(roots)
```

#img("assets/8.png", [Результат лемматизации (глагол)])

В режиме глагола слова `expanded` и `expand` были сведены к `expand`, а формы `was`, `is`, `are` к лемме `be`. Именно такие неправильные формы стеммеры привести к одному корню не способны. Формы существительных в этом режиме не были изменены, так как лемматизатор не считает их глаголами, а слова `times`, `structures`, `sets` были распознаны как формы глаголов.

Итоги лемматизации:

- результат зависит от указанной части речи. Для получения всех лемм слова необходимо обрабатывать в обоих режимах или предварительно определять часть речи каждого слова;
- лемматизатор опирается на словарь WordNet, поэтому корректно обрабатывает неправильные формы и возвращает настоящие слова;
- словообразовательные связи лемматизатор не учитывает, это делают стеммеры;
- работа лемматизатора медленнее, чем работа стеммера.

#pagebreak()
= Задание 4. Разбиение текстовых данных на информационные блоки

Были прочитаны `N` первых слов из коллекции Brown:

```py
words = brown.words()[:n]
```

Затем выборка была разбита на чанки по `M` слов:

```py
def chunk_words(words_list, chunk_size):
  for i in range(0, len(words_list), chunk_size):
    yield words_list[i : i + chunk_size]
```

```py
chunks = list(chunk_words(words, m))
```

#img("assets/9.png", [Разбитая выборка])

Из коллекции Brown были прочитаны первые `N = 1000` слов, которые были разбиты на чанки по `M = 15` слов. В результате получилось 67 чанков, из них 66 полных и один неполный, содержащий оставшиеся 10 слов. Для вывода использовались первые 20 чанков, каждый из которых был обрезан до 50 символов. Следует отметить, что чанки разбивают текст строго по числу слов и не учитывают смысл, то есть границы чанков приходятся на середину предложений. Кроме того, знаки препинания и кавычки в корпусе Brown тоже считаются словами, поэтому в чанк по 15 слов попадает меньше 15 "настоящих" слов.

#pagebreak()
= Задание 5. Извлечение частотности слов с помощью модели Bag of Words

Для выполнения задания использовались `CountVectorizer` из библиотеки `scikit-learn` и корпус Brown.

```sh
pip install scikit-learn numpy
```

```py
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import brown
```

Из коллекции Brown были прочитаны первые `N = 1000` слов, которые были разбиты на чанки по `M = 200` слов. Каждый чанк был помещен в словарь с полями `index` и `text`:

```py
words = brown.words()[:n]
chunks = [
    {"index": i, "text": " ".join(chunk)}
    for i, chunk in enumerate(chunk_words(words, m))
]
texts = [chunk["text"] for chunk in chunks]
```

Затем была построена терм-документная матрица:

```py
vectorizer = CountVectorizer()
doc_term_matrix = vectorizer.fit_transform(texts)
vocabulary = list(vectorizer.get_feature_names_out())
```

Для наглядности помимо самой матрицы была выведена таблица, содержащая 10 самых частотных слов:

```py
dense = doc_term_matrix.toarray()
totals = dense.sum(axis=0)
top_ids = np.argsort(-totals)[:10]
```

#img("assets/10.png", [Терм-документная матрица])

Была получена матрица размером 5 на 400. Анализ матрицы:

- большая часть значений матрицы равна нулю. Каждое слово встречается лишь в нескольких чанках, поэтому на практике такие матрицы хранят в разреженном виде. `fit_transform` возвращает именно разреженную матрицу, а `toarray` понадобился только для вывода;
- самыми частотными оказались служебные слова: `the`, `of`, `and`, `in`, `that`. Они встречаются во всех чанках и почти не несут информации о содержании;
- слова, зависящие от темы, распределены неравномерно. `jury` (6, 5, 4, 3, 0) и `fulton` (3, 2, 4, 5, 0) встречаются в первых 4-х чанках и отсутствуют в последнем. По таким столбцам чанки можно различать по теме;
- `CountVectorizer` приводит слова к нижнему регистру и по умолчанию игнорирует знаки препинания и односимвольные слова, поэтому словарь содержит только слова из 2-x и более букв;
- в модели Bag of Words порядок слов и грамматика не учитываются, остается только количество вхождений.

#pagebreak()
= Задание 6. Создание прогнозатора категорий

Для выполнения задания были импортированы следующие пакеты:

```py
from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
```

Были определены 5 категорий, имена которых соответствуют именам в пакете `scikit-learn`:

```py
category_map = {
    "rec.sport.hockey": "Hockey",
    "sci.space": "Space",
    "comp.graphics": "Graphics",
    "talk.politics.guns": "Politics (guns)",
    "sci.med": "Medicine",
}
```

Был получен тренировочный набор документов, а затем извлечены значения счетчиков слов и построен преобразователь TF-IDF:

```py
training_data = fetch_20newsgroups(
    subset="train",
    categories=category_map.keys(),
    shuffle=True,
    random_state=7,
)

count_vectorizer = CountVectorizer()
train_tc = count_vectorizer.fit_transform(training_data.data)

tfidf = TfidfTransformer()
train_tfidf = tfidf.fit_transform(train_tc)
```

#img("assets/11.png", [Категории классификации и размер обучающей выборки])

Для тестирования вручную были составлены 5 предложений, не входящих в тренировочный набор:

```py
input_data = [
    "The goalie made an amazing save and the team scored in overtime",
    "NASA is planning to launch a new rocket to the Moon next year",
    "I need a better video card to render 3D graphics and textures",
    "The doctor prescribed antibiotics to treat the infection",
    "The government wants to restrict the right to own a gun",
]
```

Затем был обучен мультиномиальный байесовский классификатор, а тестовые данные были преобразованы теми же `CountVectorizer` и `TfidfTransformer`:

```py
classifier = MultinomialNB().fit(train_tfidf, training_data.target)

input_tc = count_vectorizer.transform(input_data)
input_tfidf = tfidf.transform(input_tc)

predictions = classifier.predict(input_tfidf)
```

#img("assets/12.png", [Предсказанные категории тестовых предложений])

Анализ результатов:

- каждое тестовое предложение было составлено так, чтобы относиться к одной из 5-и категорий и содержать характерные для нее слова.
- классификатор опирается на слова с большим весом TF-IDF в каждой категории, поэтому уверенно определяет тему, если в предложении есть специфичные для нее слова. Общеупотребительные слова имеют низкий вес IDF и почти не влияют на результат;
- 5 выбранных категорий достаточно далеки друг от друга по тематике, что упрощает классификацию. Для близких категорий, например, `comp.graphics` и `sci.electronics` результат был бы хуже;
- классификатор работает с мешком слов, если в предложении нет ни одного слова из словаря тренировочных данных или оно состоит из общих слов, предсказание будет определяться только априорной вероятностью класса и окажется случайным.

#pagebreak()
= Задание 7. Создание анализатора грамматических родов

Для выполнения задания были импортированы следующие пакеты:

```py
from nltk import NaiveBayesClassifier
from nltk.classify import accuracy as nltk_accuracy
from nltk.corpus import names
```

Была определена функция, извлекающая последние `N` букв имени:

```py
def last_letters(word: str, num_letters: int) -> dict[str, str]:
    return {"feature": word[-num_letters:].lower()}
```

Обучающие данные были загружены из корпуса `names`, объединены в 1 список и перемешаны с фиксированным seed:

```py
male_list = [(name, "male") for name in names.words("male.txt")]
female_list = [(name, "female") for name in names.words("female.txt")]
data = male_list + female_list

random.seed(7)
random.shuffle(data)
```

Для финального тестирования вручную была составлена выборка из 8 имен:

```py
test_names = ["Anna", "Maria", "Olga", "Elena", "Nikita", "Ivan", "Dmitry", "Mark"]
```

Данные были разделены на обучающую и тестовую выборки. Для каждого `N` от 1 до 5 был обучен наивный байесовский классификатор, вычислена точность и предсказан пол для имен из ручной выборки:

```py
for n in range(1, 6):
    train_features = [(last_letters(name, n), gender) for name, gender in train_names]
    test_features = [(last_letters(name, n), gender) for name, gender in test_set]

    classifier = NaiveBayesClassifier.train(train_features)
    acc = nltk_accuracy(classifier, test_features)

    for name in test_names:
        print(f"  {name}: {classifier.classify(last_letters(name, n))}")
```

#img("assets/13.png", [Результаты для N = 1, 2, 3])

#img("assets/14.png", [Результаты для N = 4, 5 и оптимальное значение N])

Точность метода на тестовой выборке для каждого `N` приведена в таблице:

#align(center)[
  #tbl(
    [Точность классификатора в зависимости от `N`],
    columns: 6,
    inset: 6pt,
    align: center,
    [*N*],
    [*1*],
    [*2*],
    [*3*],
    [*4*],
    [*5*],
    [*Точность, %*],
    [77.16],
    [79.23],
    [77.16],
    [69.54],
    [64.19],
  )
]

Анализ результатов:

- наибольшая точность 79.23% была получена при `N = 2`, поэтому оптимальным значением является `N = 2`. При изменении seed значения могут немного отличаться, но тенденция сохраняется;
- при `N = 1` признаков слишком мало, по последней букве нельзя надежно определить пол. Например, имя `Dmitry` при `N = 1` было ошибочно классифицировано как женское, так как среди женских имен много заканчивающихся на `y`. При `N = 2` и больше оно определяется верно;
- при `N = 4` и `N = 5` точность заметно снижается. Суффиксы длиннее встречаются в обучающих данных редко и фактически повторяют целые имена. Поэтому классификатор переобучается, а на новых именах встречает суффиксы, которых не видел в обучении;
- русское мужское имя `Nikita` при `N` от 1 до 3 было определено как женское, оно оканчивается на `a`, а в англоязычном корпусе такое окончание характерно для женских имен. При `N = 4` и `N = 5` результат изменился на `male`;
- остальные имена из тестовой выборки были определены верно при любом `N`.

#pagebreak()
= Задание 8. Создание сентимент-анализатора

Для выполнения задания были импортированы следующие пакеты:

```py
from nltk.corpus import movie_reviews
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy as nltk_accuracy
```

Были загружены размеченные отзывы о фильмах, из которых извлечены признаки:

```py
fileids_pos = movie_reviews.fileids("pos")
fileids_neg = movie_reviews.fileids("neg")

def extract_features(words):
    return dict([(word, True) for word in words])

features_pos = [
    (extract_features(movie_reviews.words(fileids=[f])), "Positive")
    for f in fileids_pos
]
features_neg = [
    (extract_features(movie_reviews.words(fileids=[f])), "Negative")
    for f in fileids_neg
]
```

Данные были разбиты на тренировочный и тестовый наборы в соотношении 80/20, после чего был обучен наивный байесовский классификатор:

```py
num_pos = int(0.8 * len(features_pos))
num_neg = int(0.8 * len(features_neg))

features_train = features_pos[:num_pos] + features_neg[:num_neg]
features_test = features_pos[num_pos:] + features_neg[num_neg:]

classifier = NaiveBayesClassifier.train(features_train)
print("Accuracy (test):", nltk_accuracy(classifier, features_test))
```

#img("assets/15.png", [Метрики классификатора тональности])

Были выведены 20 наиболее информативных слов:

```py
classifier.show_most_informative_features(20)
```

#img("assets/16.png", [20 наиболее информативных слов])

Для тестирования вручную были составлены 5 предложений, похожих на отзывы о фильмах:

```py
input_reviews = [
    "The costumes in this movie were great",
    "I think the story was terrible and the characters were very weak",
    "People say that the director of the movie is amazing",
    "This is such an idiotic movie. I will not recommend it to anyone",
    "The acting was wonderful and the plot kept me interested until the very end",
]
```

Для каждого отзыва были вычислены вероятности обоих классов, и выбран класс с максимальной вероятностью:

```py
probdist = classifier.prob_classify(
  extract_features(
    nltk.word_tokenize(review.lower())
  )
)
pred_sentiment = probdist.max()
print("Predicted sentiment:", pred_sentiment)
print("Probability:", round(probdist.prob(pred_sentiment), 4))
```

#img("assets/17.png", [Предсказанная тональность тестовых отзывов])

Анализ результатов:

- точность на тестовой выборке составила 73.5%, а на обучающей 98.25%. Такая разница говорит о переобучении, классификатор запоминает слова тренировочных отзывов, в качестве признаков используются все слова без фильтрации, а на новых отзывах работает заметно хуже;
- среди наиболее информативных слов есть осмысленные, но есть и случайные слова, отражающие особенности корпуса, а не тональность, например имена и названия `anna`, `seagal`, `mulan`, а также `animators`, `annual`, `symbol`. Они дают высокое отношение вероятностей, так как встретились в нескольких отзывах одного класса;
- все 5 тестовых предложений были классифицированы верно. 3 положительных (`great`, `amazing`, `wonderful`) и 2 отрицательных (`terrible`, `idiotic`);
- вероятности для отрицательных отзывов выше, чем для положительных. Так, предложение про костюмы, где единственное сильное слово `great`, получило наименьшую уверенность. Вероятность зависит от числа слов с выраженной тональностью;
- модель не учитывает отрицания и контекст, так как слова рассматриваются независимо, поэтому фразы вида `not good` могут быть классифицированы неверно.

#pagebreak()
= Задание 9. Тематическое моделирование с использованием латентного размещения Дирихле

Для выполнения задания были импортированы следующие пакеты:

```sh
pip install gensim
```

```py
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from gensim import models, corpora
```

Текст из файла `data.txt` был разбит на предложения, токенизирован с помощью `RegexpTokenizer`, из него были удалены стоп-слова, а оставшиеся слова приведены к корневым формам стеммером `SnowballStemmer`. На основе полученных токенов были созданы словарь и терм-документная матрица:

```py
tokenizer = RegexpTokenizer(r"\w+")
stop_words = set(stopwords.words("english"))
stemmer = SnowballStemmer("english")

tokens = []
for line in TEXT_INPUT.split("\n"):
    if not line.strip():
        continue
    line_tokens = tokenizer.tokenize(line.lower())
    line_tokens = [t for t in line_tokens if t not in stop_words]
    line_tokens = [stemmer.stem(t) for t in line_tokens]
    tokens.append(line_tokens)

dict_tokens = corpora.Dictionary(tokens)
doc_term_mat = [dict_tokens.doc2bow(token) for token in tokens]
```

Входной текст содержит две темы, поэтому была построена модель LDA с двумя темами:

```py
ldamodel = models.ldamodel.LdaModel(
    doc_term_mat,
    num_topics=2,
    id2word=dict_tokens,
    passes=25,
    random_state=42,
)

for topic_id in range(2):
    for word, weight in ldamodel.show_topic(topic_id, topn=5):
        print(f"  {word}: {round(float(weight), 4)}")
```

#img("assets/18.png", [Представительные слова тем и распределение тем по предложениям])

Представительные слова каждой темы и их вклады:

#align(center)[
  #tbl(
    [Пять представительных слов для каждой темы],
    columns: 4,
    inset: 6pt,
    align: center,
    [*Тема 0*],
    [*Вклад*],
    [*Тема 1*],
    [*Вклад*],
    [`empir`],
    [0.0352],
    [`mathemat`],
    [0.0379],
    [`formul`],
    [0.0321],
    [`set`],
    [0.0271],
    [`time`],
    [0.0321],
    [`structur`],
    [0.0271],
    [`histor`],
    [0.0209],
    [`call`],
    [0.0271],
    [`peopl`],
    [0.0209],
    [`later`],
    [0.0162],
  )
]

Тема 0 по словам `empir`, `histor`, `peopl` соответствует истории, а тема 1 по словам `mathemat`, `set`, `structur` математике. Из 10 предложений модель верно отнесла к теме 7, предложения 1, 4 и 9 попали в тему 0, а 2, 6, 7 и 10 в тему 1. Ошибочно были классифицированы предложение 3, предложение 5 и предложение 8.

Причины ошибок связаны с малым объемом данных. Корпус состоит всего из 10 коротких предложений и содержит небольшой словарь, поэтому значения вкладов слов невелики, а несколько общих слов попали в число представительных. Слово `formul` встречается как в историческом, так и в математическом контексте, из-за чего предложение 5 оказалось отнесено к теме 0. Модель LDA является вероятностной, поэтому при смене параметра `random_state` результат может немного измениться.

#pagebreak()
= Задание 10. Самостоятельное решение задачи NLP с помощью библиотеки Hugging Face

Для выполнения задания были установлены следующие зависимости:

```sh
pip install transformers datasets torch accelerate scikit-learn
```

== Описание задачи NLP

В качестве задачи был выбран анализ тональности. Бинарная классификация текстов отзывов на фильмы на положительные и отрицательные. Эта задача схожа с заданием 8, но здесь вместо наивного байесовского классификатора на мешке слов используется предобученная модель-трансформер, которая учитывает контекст слов.

== Описание датасета

Был использован датасет IMDB из библиотеки Hugging Face. Он содержит 25 000 обучающих и 25 000 тестовых отзывов, классы сбалансированы. Для ускорения обучения были взяты подвыборки, 4000 случайных обучающих отзывов, из которых 3600 использовались для обучения и 400 для валидации, и 1000 случайных отзывов из тестовой части.

```py
dataset = load_dataset("imdb")
train_full = dataset["train"].shuffle(seed=SEED).select(range(4000))
test_ds = dataset["test"].shuffle(seed=SEED).select(range(1000))

split = train_full.train_test_split(test_size=0.1, seed=SEED)
train_ds, val_ds = split["train"], split["test"]
```

#img("assets/19.png", [Описание датасета])

Предобработка выполнялась токенизатором `AutoTokenizer` для выбранной модели. Тексты были усечены до 256 токенов, а дополнение до одной длины внутри батча выполняется автоматически в `Trainer`:

```py
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, max_length=256)

train_ds = train_ds.map(tokenize, batched=True)
```

== Описание модели и процедуры ее дообучения

Была выбрана модель `distilbert-base-uncased`, представляющая собой уменьшенную версию BERT. В ней 6 слоев энкодера вместо 12, а работает она быстрее примерно на 60% при сохранении большей части качества. Поверх энкодера добавляется классификационная голова на 2 класса.

Дообучение выполнялось с помощью класса `Trainer` со следующими параметрами: 2 эпохи, размер батча 16, скорость обучения 2e-5, weight decay 0.01, оценка на валидационной выборке после каждой эпохи, фиксированный seed. Метрики Accuracy и F1-score считались с помощью `scikit-learn`.

```py
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=2, id2label=id2label, label2id=label2id
)

args = TrainingArguments(
    output_dir="results",
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    eval_strategy="epoch",
    seed=SEED,
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
```

#img("assets/20.png", [Процесс дообучения модели])

== Метрики валидации

```py
val_metrics = trainer.evaluate(val_ds)
```

#img("assets/21.png", [Метрики на валидационной выборке])

На валидационной выборке были получены значения Accuracy = 0.8675 (86.75%) и F1-score = 0.8753. Для сравнения, после первой эпохи эти значения составляли 0.85 и 0.8678, то есть вторая эпоха улучшила качество.

== Метрики теста

```py
test_metrics = trainer.evaluate(test_ds)
```

#img("assets/22.png", [Метрики на тестовой выборке])

На тестовой выборке были получены значения Accuracy = 0.88 (88%) и F1-score = 0.8812.

== Тестирование на произвольных текстах

Модель была протестирована на 6 текстах, не входящих в выборку, с помощью `pipeline("text-classification")`. Среди них есть явно положительные и отрицательные отзывы, а также нейтральные и неоднозначные формулировки:

```py
custom_texts = [
    "This movie was absolutely fantastic, I loved every minute of it!",
    "A complete waste of time. The plot made no sense and the acting was awful.",
    "The film is fine, nothing special, but not bad either.",
    "I expected much more from this director, what a disappointment.",
    "Brilliant performances and a touching story, highly recommended.",
    "Not good. Not bad. Just boring.",
]
```

#img("assets/23.png", [Предсказания модели на произвольных текстах])

== Анализ результатов

- модель семейства BERT после всего 2-x эпох дообучения на 3600 отзывах показала Accuracy 88% и F1-score 0.8812 на тесте, что заметно выше, чем у наивного байесовского классификатора из задания 8. Преимущество объясняется тем, что трансформер использует предобученные представления слов и учитывает контекст, а не только наличие отдельных слов;
- значения Accuracy и F1-score близки друг к другу, так как классы в датасете сбалансированы;
- близость Accuracy на валидационной и тестовой выборках говорит об отсутствии сильного переобучения. При этом на 2-й эпохе потери на обучении продолжили падать, а валидационные потери немного выросли с 0.3145 до 0.3252, поэтому при увеличении числа эпох переобучение стало бы заметным;
- все 4 явно окрашенных отзыва были классифицированы верно и с высокой уверенностью. Положительные первый и пятый со score 0.9757 и 0.9733, отрицательные второй и четвертый со score 0.9822 и 0.9445. Нейтральный третий текст `nothing special, but not bad either` был отнесен к `NEGATIVE` с наименьшей уверенностью 0.8097. В обучающих данных нет нейтрального класса, поэтому модель вынуждена выбрать одну из двух меток. Неоднозначный шестой текст `Not good. Not bad. Just boring.` модель уверенно назвала отрицательным, ориентируясь на слово `boring`, что в целом соответствует смыслу;
- метрики можно улучшить, увеличив число обучающих примеров, число эпох или длину последовательности, либо использовав более крупную модель.

#pagebreak()
= Репозиторий

Исходные коды всех заданий доступны по #link("https://github.com/fymious/modern-ai/tree/main/lab2/")[ссылке].
