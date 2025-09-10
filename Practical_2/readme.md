## Word2Vec Training & Evaluation on Alice’s Adventures in Wonderland

This project implements a custom Word2Vec model trained on Lewis Carroll's "Alice’s Adventures in Wonderland" to explore how neural word embeddings capture semantic relationships in literary text.

### What This Code Does

- **Loads and preprocesses** the Alice in Wonderland text

- **Trains a Word2Vec model** to learn word relationships from the book

- **Evaluates the model** on word similarities and analogies

- **Visualizes learned relationships** to see how characters and concepts cluster together

The model represents each word as a vector where semantically similar words (like "alice" and "girl") have similar vector representations.


### Key Features

**Data Preprocessing**

- Text cleaning (URLs, punctuation, numbers removal)

- Tokenization into sentences/words

- Configurable preprocessing options

**Model Training**

- Supports both CBOW and Skip-gram architectures

- Automatic parameter recommendation based on corpus size

- Progress logging during training

- Model saving and loading


**Evaluation & Analysis**

- Word similarity evaluation

- Analogy solving capabilities

- Vocabulary coverage analysis

- Multiple visualization methods**

**Visualization Tools**

- 2D word embedding plots using t-SNE/PCA
- Model performance dashboards
- Word frequency analysis
- Similarity distribution plots


### Evaluation  Results

**My Progress Journey**  

I started with 0% accuracy on analogy tasks and improved to 25%. Here's what I learned :

```
Initial Results:
Warning: Too few valid word pairs for reliable evaluation
Analogy Evaluation:
Valid analogies: 4
Correct predictions: 1
Accuracy: 0.2500


```
- Valid analogies: 4 - **Only 4 out of the test analogies had all words in my vocabulary**
- Correct predictions: 1 - **My model got 1 analogy completely right**
- Accuracy: 0.2500 - 25% accuracy, which is actually good progress!


**Why Evaluation Was Challenging** 

- Missing words - My text didn't have all the words needed for standard tests (like "king", "queen", "man", "woman")
- Small text file - Not enough examples to learn strong word relationships
- Different topic - Standard tests use general words, but my text might be about specific topics that don't match





### What I Learned

Training Word2Vec on Alice in Wonderland taught me:

- How literary characters can be mathematically represented
- Why domain-specific models sometimes perform differently on standard tests
- The value of visualization in understanding word relationships
- That 25% accuracy represents real learning, not failure
- How preprocessing affects model quality

The visualizations were really helpful - I could actually see which words were similar to each other in the plots.
