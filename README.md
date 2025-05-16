# 📄 **Text Summarization Model Builder**

This repository contains the **model training pipeline** for a **text summarization project** using the T5 Transformer model and **readability scoring** using three machine learning models: **Ridge Regression**, **Gradient Boosting**, and **Random Forest**.

---

## ✨ **Overview**

This project focuses on:
- Fine-tuning the **T5 Transformer** model for summarization tasks.
- Training **readability scoring models** to predict the readability level of summarized text.
- Exporting trained models as `.joblib` files for easy integration into other applications, like the Django-based **Text Summarization Web App** (hosted separately).

---

## 📂 **Project Structure**

| **File/Folder**                    | **Description**                                                                                     |
|-------------------------------------|-----------------------------------------------------------------------------------------------------|
| `t5_fine_tuned/`                   | Folder containing the **fine-tuned T5 model** files like `pytorch_model.bin` and tokenizer files.    |
| `train.csv`                        | Training dataset for the summarization model (large file).                                          |
| `test.csv`                         | Testing dataset for evaluation of the trained summarization model.                                 |
| `subset_data.csv`                  | Smaller subset of the training data used for quick experimentation.                                |
| `cleaned_subset_data.csv`          | Preprocessed version of `subset_data.csv`.                                                         |
| `Summarization_code_with_T5_Model.ipynb` | Jupyter notebook for fine-tuning the T5 model on summarization tasks.                              |
| `Readability_score_with_3_models.ipynb` | Notebook for training readability scoring models (Ridge, Gradient Boosting, Random Forest).         |
| `gradient_boosting_pipeline.joblib`| Trained **Gradient Boosting** model for readability scoring.                                        |
| `ridge_regression_pipeline.joblib` | Trained **Ridge Regression** model for readability scoring.                                         |
| `random_forest_pipeline.joblib`    | Trained **Random Forest** model for readability scoring.                                            |
| `tfidf_vectorizer.pkl`             | Vectorizer used to transform text features for the readability scoring models.                     |
| `linguistic_features.py`           | Script to extract linguistic features (like word count, sentence count) from text data.            |

---

## 💻 **How It Works**

### 1️⃣ **Fine-Tuning T5 for Summarization**
- The T5 model is fine-tuned using the **Hugging Face Transformers** library.
- Preprocessed training data is used for summarization.
- The model is saved to the `t5_fine_tuned/` folder for future use.

### 2️⃣ **Readability Scoring**
- **Three models** are trained to predict readability:
  - **Ridge Regression**
  - **Gradient Boosting**
  - **Random Forest**
- Each model uses text features extracted using `tfidf_vectorizer.pkl` and linguistic properties from `linguistic_features.py`.
- Trained models are saved as `.joblib` files.

### 3️⃣ **Data Preprocessing**
- Raw datasets (`train.csv`, `subset_data.csv`) are cleaned and saved as `cleaned_subset_data.csv`.
- Preprocessing includes handling missing values, tokenization, and feature extraction.

---

## ⚙️ **Steps to Reproduce**

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Devbhagat718/text-summarization-model-builder.git
   cd text-summarization-model-builder
   ```

2. **Launch Jupyter Notebooks**
   - Open the `Summarization_code_with_T5_Model.ipynb` notebook to train or fine-tune the summarization model.
   - Open `Readability_score_with_3_models.ipynb` to train the readability scoring models.

3. **Load Data**
   - Ensure `train.csv` and `test.csv` are available in the same directory.

4. **Export Models**
   - Fine-tuned T5 models and readability scoring models are saved as `.joblib` files for future integration.

---

## 🌟 **Key Features**

- **Summarization**: Leverages the T5 Transformer for high-quality summarization.
- **Readability Scoring**: Predicts text readability using multiple machine learning models.
- **Preprocessing**: Provides cleaned and feature-enriched datasets for training.

---

## 🚀 **Future Enhancements**

- Expand datasets for better generalization.
- Incorporate more advanced readability scoring metrics.
- Optimize summarization for different languages and domains.

---

## 🔗 **Related Repository**
The **Text Summarization Web Application**, which uses the models trained in this project, can be found here:  
[🌐 Text Summarization Web App](https://github.com/Devbhagat718/text-summarizer-web-app)

---

## 📜 **License**

This project is licensed under the [MIT License](LICENSE).

---
