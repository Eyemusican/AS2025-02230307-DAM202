# Text Preprocessing Pipeline for NLP

## What is This Project?

This project cleans messy text data and prepares it for machine learning. I built a system that takes raw text (with emojis, punctuation, URLs, etc.) and converts it into clean, organized data that computers can understand.

**Main Goal:** Build a text cleaning pipeline that works on any text data and turns it into numbers for machine learning.

**What This Does:**
-  Makes all text lowercase
-  Removes junk (URLs, emails, special characters)
-  Splits text into individual words
-  Converts words to their simple forms (running → run)
-  Removes common words like "the", "and", "is"
-  Turns text into numbers for machine learning

---

## My Implementation

### Step 0: Creating Sample Data

First, I made a sample dataset to test my pipeline.

**My Code:**
```python
import pandas as pd

data = [
    "When life gives you lemons, make lemonade! 🙂",
    "She bought 2 lemons for $1 at Maven Market.",
    "A dozen lemons will make a gallon of lemonade. [AllRecipes]",
    "lemon, lemon, lemons, lemon, lemon, lemons",
    "He's running to the market to get a lemon — there's a great sale today.",
    "Does Maven Market carry Eureka lemons or Meyer lemons?",
    "An Arnold Palmer is half lemonade, half iced tea. [Wikipedia]",
    "iced tea is my favorite"
]

# Convert to DataFrame
data_df = pd.DataFrame(data, columns=['sentence'])
pd.set_option('display.max_colwidth', None)
```

**What it does:**
- Makes a list of 8 sentences with different problems (emojis, numbers, punctuation)
- Turns the list into a table format using pandas
- Shows full text without cutting it off

**Why this is important:** Real text is messy! This dataset has the same problems as real social media posts or reviews.

---

### Step 1: Making Text Lowercase

I converted all text to lowercase so "Lemon" and "lemon" are treated the same.

**My Code:**
```python
# Make a copy to work with
spacy_df = data_df.copy()

# Change everything to lowercase
spacy_df['clean_sentence'] = spacy_df['sentence'].str.lower()
```

**What it does:**
- Keeps a copy of original data
- Changes ALL letters to lowercase
- Saves result in a new column

**Example:**
- Before: "When life gives you lemons, make lemonade! 🙂"
- After: "when life gives you lemons, make lemonade! 🙂"

**Why this matters:** Without this, "Lemon", "lemon", and "LEMON" would be seen as different words.

---

### Step 2: Removing Unwanted Stuff

This step removes URLs, emails, hashtags, and special characters.

**My Code:**
```python
# Remove citations like [wikipedia]
spacy_df['clean_sentence'] = spacy_df['clean_sentence'].str.replace('[wikipedia]', '')

# Remove everything we don't need using regex
combined = r'https?://\S+|www\.\S+|<.*?>|\S+@\S+\.\S+|@\w+|#\w+|[^A-Za-z0-9\s]'
spacy_df['clean_sentence'] = spacy_df['clean_sentence'].str.replace(combined, ' ', regex=True)

# Fix spacing (remove extra spaces)
spacy_df['clean_sentence'] = spacy_df['clean_sentence'].str.replace(r'\s+', ' ', regex=True).str.strip()
```

**What it does:**
- Removes website links (http://...)
- Removes emails (user@email.com)
- Removes @ mentions and # hashtags
- Removes punctuation and emojis
- Removes numbers and special symbols
- Fixes extra spaces

**Example:**
- Before: "she bought 2 lemons for $1 at maven market."
- After: "she bought 2 lemons for 1 at maven market"

**Why this matters:** Machine learning works better with clean text. Removing junk helps the computer focus on actual words.

---

### Step 3: Using spaCy for Smart Text Processing

I used spaCy, a smart text processing tool, for advanced cleaning.

**My Code:**
```python
import spacy

# Load English language tool
nlp = spacy.load('en_core_web_sm')

# Process a sentence
phrase = spacy_df.clean_sentence[0]  # "when life gives you lemons make lemonade"
doc = nlp(phrase)
```

**What it does:**
- Loads a tool that understands English grammar
- Processes text to understand words and their meanings
- Creates an object with all word information

**Why this matters:** spaCy is smarter than simple text cleaning - it understands how words work in English.

---

### Step 3.1: Breaking Text into Words

This splits sentences into individual words.

**My Code:**
```python
# Get individual words
tokens = [token.text for token in doc]
# Result: ['when', 'life', 'gives', 'you', 'lemons', 'make', 'lemonade']
```

**What it does:**
- Takes the sentence
- Splits it into separate words
- Makes a list of words

**Why this matters:** To analyze text, we need to look at individual words first.

---

### Step 3.2: Converting Words to Base Forms

This changes words to their dictionary form.

**My Code:**
```python
# Get base forms of words
lemmas = [token.lemma_ for token in doc]
# Result: ['when', 'life', 'give', 'you', 'lemon', 'make', 'lemonade']
```

**What it does:**
- Changes "gives" → "give"
- Changes "lemons" → "lemon"
- Changes "running" → "run"
- Changes "better" → "good"

**Example changes:**
- "was" → "be"
- "mice" → "mouse"
- "children" → "child"

**Why this matters:** This groups similar words together. "run", "running", "ran" all become "run", so the computer knows they mean the same thing.

---

### Step 3.3: Removing Common Words

Common words like "the", "and", "is" don't add much meaning, so I remove them.

**My Code:**
```python
# See all common words
stop_words = list(nlp.Defaults.stop_words)
print(f"Total stop words: {len(stop_words)}")  # 326 words

# Remove common words
filtered = [token for token in doc if not token.is_stop]
# Result: [life, gives, lemons, lemonade]

# Get base forms AND remove common words
cleaned = [token.lemma_ for token in doc if not token.is_stop]
# Result: ['life', 'give', 'lemon', 'lemonade']

# Put words back into sentence
result = ' '.join(cleaned)
# Result: 'life give lemon lemonade'
```

**What it does:**
- Shows there are 326 common English words
- Removes words like "when", "you", "the", "and"
- Keeps only meaningful words
- Combines with word base forms

**Why this matters:** Common words appear everywhere but don't tell us much. Removing them lets us focus on important words.

---

### Step 4: Making Reusable Functions

I made functions so I can reuse this code easily.

**My Code:**
```python
def token_lemma_stopw(text):
    """
    Cleans text by converting to base forms and removing common words.
    """
    doc = nlp(text)
    output = [token.lemma_ for token in doc if not token.is_stop]
    return ' '.join(output)

# Use on all sentences
spacy_df['processed'] = spacy_df.clean_sentence.apply(token_lemma_stopw)
```

**What it does:**
- Wraps cleaning steps in one function
- Takes any text and cleans it
- Can use on one sentence or many sentences

**Why this matters:** Functions save time - write once, use many times.

---

### Step 5: Complete Cleaning Pipeline

I put all cleaning steps together in one pipeline.

**My Code:**
```python
def lower_replace(series):
    """Makes text lowercase and removes junk."""
    output = series.str.lower()
    combined = r'https?://\S+|www\.\S+|<.*?>|\S+@\S+\.\S+|@\w+|#\w+|[^A-Za-z0-9\s]'
    output = output.str.replace(combined, ' ', regex=True)
    return output

def nlp_pipeline(series):
    """
    Complete cleaning pipeline.
    
    Steps:
    1. Make lowercase
    2. Remove junk with regex
    3. Convert to base word forms
    4. Remove common words
    """
    output = lower_replace(series)
    output = output.apply(token_lemma_stopw)
    return output

# Clean all data
cleaned_text = nlp_pipeline(data_df.sentence)

# Save cleaned data
pd.to_pickle(cleaned_text, 'preprocessed_text.pkl')
```

**What it does:**
- Combines ALL cleaning steps
- One function call does everything
- Saves cleaned data for later use

**Cleaning flow:**
Raw text → Lowercase → Remove junk → Split into words → Base forms → Remove common words → Clean text

**Why this matters:** One function does all cleaning consistently on any text.

---

### Step 6: Turning Text into Numbers (Bag of Words)

Computers need numbers, not words. I converted text to a number table.

**My Code:**
```python
from sklearn.feature_extraction.text import CountVectorizer

# Load cleaned text
series = pd.read_pickle('preprocessed_text.pkl')

# Convert text to numbers
cv = CountVectorizer()
bow = cv.fit_transform(series)

# Make it readable
bow_df = pd.DataFrame(bow.toarray(), columns=cv.get_feature_names_out())
```

**What it does:**
- Loads cleaned text
- Counts how many times each word appears in each sentence
- Makes a table where:
  - Each row = one sentence
  - Each column = one unique word
  - Each cell = how many times that word appears

**Example table:**

| Sentence | lemon | life | give | tea | market |
|----------|-------|------|------|-----|--------|
| 0        | 2     | 1    | 1    | 0   | 0      |
| 1        | 1     | 0    | 0    | 0   | 1      |

**Why this matters:** Machine learning needs numbers. This table shows word counts for each sentence.

---

### Step 6.1: Better Word Counting

I added filters to get better quality word counts.

**My Code:**
```python
# Count words with filters
cv1 = CountVectorizer(
    stop_words='english',      # Remove common words automatically
    ngram_range=(1, 1),         # Count single words only
    min_df=2                    # Only count words in 2+ sentences
)

bow1 = cv1.fit_transform(series)
bow1_df = pd.DataFrame(bow1.toarray(), columns=cv1.get_feature_names_out())

# Find most common words
term_freq = bow1_df.sum()
print(term_freq.sort_values(ascending=False))
```

**What it does:**
- Automatically removes common words like "the", "and"
- Only counts single words (not phrases)
- Ignores words that appear in only 1 sentence (probably mistakes)
- Shows which words appear most often

**Why this matters:** Filters remove noise and focus on important words that appear multiple times.

---

### Step 7: TF-IDF (Smart Word Weighting)

TF-IDF is smarter than simple counting - it considers word importance.

**My Code:**
```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Basic TF-IDF
tv = TfidfVectorizer()
tfidf = tv.fit_transform(series)
tfidf_df = pd.DataFrame(tfidf.toarray(), columns=tv.get_feature_names_out())

# TF-IDF with filters
tv1 = TfidfVectorizer(min_df=2)
tfidf1 = tv1.fit_transform(series)
tfidf1_df = pd.DataFrame(tfidf1.toarray(), columns=tv1.get_feature_names_out())
```

**What it does:**
- Calculates importance scores instead of just counting
- Gives high scores to words that are:
  - Common in one sentence (appears a lot)
  - Rare across all sentences (unique/special)

**Example:**
- Word "lemon" in 6/8 sentences → Lower score (too common)
- Word "gallon" in only 1 sentence → Higher score (unique/special)

**Scores mean:**
- Close to 1 = very important word for that sentence
- Close to 0 = common word or not in sentence

**Why this matters:** TF-IDF finds the most important/unique words for each sentence, not just frequent words.

---

### Step 7.1: Counting Word Pairs

I also counted word pairs (bigrams) to capture meaning better.

**My Code:**
```python
# Count single words AND word pairs
tv2 = TfidfVectorizer(ngram_range=(1, 2))  # 1 word + 2 word pairs
tfidf2 = tv2.fit_transform(series)
tfidf2_df = pd.DataFrame(tfidf2.toarray(), columns=tv2.get_feature_names_out())

# Find most important words/phrases
importance = tfidf2_df.sum().sort_values(ascending=False)
print(importance.head(10))
```

**What it does:**
- Counts both single words: "lemon", "tea"
- AND word pairs: "ice tea", "arnold palmer"
- Calculates importance for all of them

**Word pairs captured:**
- "ice tea" (not just "ice" and "tea" separately)
- "arnold palmer" (the drink name)
- "maven market" (the store name)

**Why this matters:** Word pairs keep meaning:
- "not good" (negative) vs just "good" (positive)
- "ice tea" (specific drink) vs "ice" + "tea" (separate things)

---

## What I Learned

### 1. **The Cleaning Process**
Raw text → Lowercase → Remove junk → Split words → Base forms → Remove common words → Turn into numbers

Each step makes the text cleaner and easier for computers to understand.

### 2. **Why Each Step is Important**
- **Lowercase:** Groups "Lemon" and "lemon" as same word
- **Remove junk:** Gets rid of noise
- **Split words:** Analyzes text word by word
- **Base forms:** Groups "run", "running", "ran" together
- **Remove common words:** Focuses on meaningful words
- **Turn to numbers:** Computers need numbers, not text

### 3. **Counting vs Smart Weighting**
- **Simple counting (BoW):** Just counts words
- **Smart weighting (TF-IDF):** Finds important words
- **When to use counting:** Simple tasks
- **When to use TF-IDF:** Better for most tasks

### 4. **Single Words vs Word Pairs**
- **Single words:** Faster but loses context
- **Word pairs:** Keeps some meaning but bigger data
- **Example:** "not good" means something different than just "good"

### 5. **Reusable Code**
Making functions helps me:
- Use same cleaning on different data
- Save time
- Make fewer mistakes


## What This is Used For

This cleaning pipeline works for:
- **Sentiment Analysis:** Is a review positive or negative?
- **Spam Detection:** Is this email spam?
- **Text Classification:** What category is this article?
- **Topic Discovery:** What topics are in these documents?
- **Search Engines:** Finding relevant documents
- **Chatbots:** Understanding what users type


## Summary

Through this project, I learned:
- How to clean messy text data
- Using regular expressions to remove junk
- Using spaCy for smart text processing
- Converting text to numbers for machine learning
- Difference between simple counting and smart weighting
- Building reusable cleaning pipelines

