# Spoiler Detector for Goodreads Reviews

Being an avid reader and a frequent Goodreads user to look up reviews and ratings for my next read, I find it frustrating to come across spoilers. This idea inspired me to build a spoiler detection system for Goodreads reviews.

## References
I used the following research paper as a reference for this project:  
**Spoiler Alert: Using Natural Language Processing to Detect Spoilers in Book Reviews** by Bao et al., 2021.

## Dataset Details
- **Source:** Kaggle  
- **Books:** 25,475  
- **Users:** 18,892  
- **Reviews:** 1,378,033  

---

## Libraries Used
- **Pandas:** Reading, exploring, and cleaning the data.  
- **Scikit-learn:** Machine learning pipeline, data splitting, feature extraction, model training, and evaluation.  
- **NLTK (Natural Language Toolkit):** Text preprocessing.  
- **Matplotlib & NumPy:** Visualization.  

---

## Workflow

### 1. Data Loading
The dataset was loaded from JSON format into a Pandas DataFrame, where each row represents a review along with metadata such as user ID, timestamp, and spoiler flag.  
- Handled large datasets and inspected the dataset structure using methods like `df.info()` and `df.head()`.

---

### 2. Data Exploration
- Analyzed the class distribution with respect to the number of spoiler vs. non-spoiler reviews.

---

### 3. Data Cleaning and Preprocessing
Performed overall noise reduction, including:  
- **Punctuation Removal:** Removed unnecessary characters.  
- **Lowercasing:** Converted all text to lowercase for uniformity.  
- **Stopword Removal:** Filtered out common, unimportant words using NLTK's list of stopwords.  

---

### 4. Data Splitting
- Split the dataset into training and testing sets using the `train_test_split` function.  
- Used an 80% training and 20% testing setup.

---

### 5. Text Vectorization
Since machine learning models cannot directly interpret text, the cleaned text was converted into numerical vectors. This was achieved using a **TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer**, which transforms the text into a sparse matrix representing word importance with respect to the entire text.

---

### 6. Model Training
A **Logistic Regression** model was trained on the vectorized text data to predict whether a review contains spoilers. Logistic regression was chosen because it is well-suited for binary classification tasks.

---

## Metrics
- **Accuracy:** The percentage of correct predictions.  
- **Precision:** Measures how many predicted spoiler reviews are actually spoilers.  
- **Recall:** Measures how many actual spoilers are correctly predicted.  
- **F1-Score:** The harmonic mean of precision and recall.  

---

## Results and Problems

### Results:
- The model achieves an overall accuracy of **93.6%**.  

### Problems:
- **Class Imbalance:**  
   - While the accuracy is high, the model struggles with detecting spoiler reviews.  
   - It achieves only **55% precision** and **9% recall** for the "spoiler" class due to the imbalance in the dataset.  
- **Imbalance Handling:**  
   - Techniques like **undersampling** or **SMOTE (Synthetic Minority Oversampling Technique)** could address this issue. However, SMOTE was not feasible due to the large dataset size and resampling challenges.

---

## Future Developments

1. **SVM Implementation:**  
   - Experiment with Support Vector Machines (SVM) to improve classification.
   
2. **Feature Engineering:**  
   - Add features like **Genre**, **Rating**, and **User Contribution** to better assess the likelihood of spoilers.

3. **N-grams:**  
   - Extend TF-IDF to include **bigrams** or **trigrams** to capture more context rather than relying only on unigrams.

4. **Word Embeddings:**  
   - Use pre-trained word embeddings like **Word2Vec** or **GloVe** to capture semantic similarities between words.

5. **Transformer Models:**  
   - Experiment with **BERT (Bidirectional Encoder Representations from Transformers)** for better sentence understanding. Huggingface’s transformers library could be used for this.

6. **Alternative Classifiers:**  
   - Test advanced models like **Random Forest** or **XGBoost** instead of Logistic Regression.

---
