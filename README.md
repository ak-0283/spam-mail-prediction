# 📧 Spam Mail Prediction using **Logistic Regression**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-username/spam-mail-prediction/blob/main/Spam_Mail_Prediction.ipynb)

This is my **third Machine Learning project** after learning from [Siddhardhan's YouTube channel](https://www.youtube.com/@Siddhardhan) 🚀.
The project predicts whether an email is **Spam (0)** or **Ham (1)** using the **Logistic Regression** algorithm.

---

## 📚 Learning Journey

* Followed Siddhardhan's tutorials on YouTube to learn **ML fundamentals**.
* Implemented the project using **Google Colab**.
* Learned about **data preprocessing, TF-IDF feature extraction, Logistic Regression model building, and evaluation**.

---

## 📊 Dataset Information

* **Shape:** (5572, 2)
* **Target Variable:**

  * `0` → Spam
  * `1` → Ham

> Note: The dataset is included in this repository in the dataset.txt file. Please download it manually or you can use a dataset from **Kaggle** or **UCI ML Repository** 👍.

---

## 🛠️ Libraries Used

* Python
* Google Colab
* NumPy
* Pandas
* scikit-learn (TfidfVectorizer, train\_test\_split, LogisticRegression, accuracy\_score)

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
```

---

## 🔄 Project Workflow

1. **Mail Data** → Load dataset for analysis
2. **Data Preprocessing** → Encode labels (Spam = 0, Ham = 1)
3. **Train-Test Split** → Separate data into training & testing sets
4. **Feature Extraction** → Convert text into numerical features using `TfidfVectorizer`
5. **Model Building** → Logistic Regression classifier
6. **Model Evaluation** → Accuracy on training & testing sets

---

## 📊 Data Splits & Accuracy

* **Algorithm:** Logistic Regression
* **Training Accuracy:** 96.70%
* **Testing Accuracy:** 96.59%

```python
print('Accuracy on training data : ', accuracy_on_training_data)
# Accuracy on training data :  0.9670181736594121

print('Accuracy on test data : ', accuracy_on_test_data)
# Accuracy on test data :  0.9659192825112107
```

---

## 💻 How to Run

1. **Clone the repository**

```bash
git clone https://github.com/your-username/spam-mail-prediction.git
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Run the notebook in Google Colab** using the badge above or run locally:

```bash
jupyter notebook notebooks/Spam_Mail_Prediction.ipynb
```

---

## 🎯 Conclusion

This project helped me:

* Understand **text preprocessing** with TF-IDF
* Learn **Logistic Regression** for classification tasks
* Evaluate model performance on training & testing sets

It was another exciting step in my **Machine Learning journey**! 🚀

---

## 🙌 Acknowledgments

Special thanks to **Siddhardhan** for his beginner-friendly ML tutorials on YouTube. ⭐ If you found this helpful, consider giving this repo a star!

---
