# 💊 Prescription Parser: Medical NER with CRF

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![Library](https://img.shields.io/badge/Lib-sklearn--crfsuite-orange) ![NLP](https://img.shields.io/badge/Task-Named%20Entity%20Recognition-green)

### 🏥 Project Overview
In the healthcare industry, prescriptions are often written as unstructured text. Converting these instructions into structured digital records (compatible with standards like **FHIR**) is essential for patient safety and pharmacy automation.

This project builds a **Named Entity Recognition (NER)** system using **Conditional Random Fields (CRF)**. It takes a raw prescription string and automatically "tags" each word with its specific medical function.

---

### 🏷️ The Challenge: Sequence Prediction
Unlike standard classification, predicting tags for a sentence requires understanding context (e.g., "2" could be a dosage or a duration depending on the words around it). 

**Input:** `"Take 2 tablets once a day for 10 days"`

**Output (Predicted Labels):**
* **Method:** `Take`
* **Qty:** `2`
* **Form:** `tablets`
* **Frequency:** `once`
* **Period:** `a day`
* **Duration:** `10 days`

---

### 🧠 Methodology
I utilized **Conditional Random Fields (CRF)**, a probabilistic framework that is superior to standard classifiers for sequence data because it considers the "neighboring" labels when making a prediction.

1.  **Data Preprocessing:** Tokenized sentences into individual words.
2.  **Feature Engineering:** Extracted features for each word to help the model context:
    * Is the word a number?
    * Is it a suffix/prefix?
    * What is the previous word? (Context window)
    * Is it capitalized?
3.  **Model Training:** Trained the `sklearn-crfsuite` model on labeled medical instruction data.
4.  **Evaluation:** Measured performance using the **Flat F1-Score** and Classification Report to ensure accurate extraction across all tag types.

---

### 🛠️ Setup & Usage
1.  **Install Dependencies:**
    ```bash
    pip install sklearn-crfsuite pandas numpy scikit-learn
    ```
2.  **Run the Parser:**
    Open `Karthik_Task1_Prescription_Parser.ipynb` in Jupyter Notebook.
3.  **Data:**
    The training data is generated/loaded directly within the notebook for demonstration.

---

### 👨‍💻 About the Author
**Karthik Kunnamkumarath**
*Aerospace Engineer | Project Management Professional (PMP) | AI Solutions Developer*

I combine engineering precision with data science to solve complex problems.
* 📍 Toronto, ON
* 💼 [LinkedIn Profile](https://linkedin.com/in/4karthik95)
* 📧 Aero13027@gmail.com

---

### 💻 Code Snippet: Predicting Tags
Here is how the trained CRF model predicts tags for a new prescription:

```python
import sklearn_crfsuite

# Example input sentence (tokenized)
sentence = ["Take", "2", "tabs", "every", "6", "hours"]

# Extract features for the sentence (using helper function defined in notebook)
features = [word2features(sentence, i) for i in range(len(sentence))]

# Predict labels
labels = crf.predict_single(features)

# Display Result
for word, label in zip(sentence, labels):
    print(f"{word}: {label}")

# Output:
# Take: Method
# 2: Qty
# tabs: Form
# every: Frequency
# 6: Period
# hours: PeriodUnit
