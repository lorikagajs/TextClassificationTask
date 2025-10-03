# Multi-Label Emotion Classification from Text

This project is a **multi-label text classification system** that predicts multiple emotions from raw text data. It includes **preprocessing, feature extraction, exploratory data analysis (EDA), model training, evaluation, and saving trained models**.  

The models used are:  
- **Logistic Regression** (wrapped in One-vs-Rest)  
- **Multinomial Naive Bayes** (wrapped in One-vs-Rest)  

---

## 📄 Dataset

The dataset used in this project is taken from the following paper:

> SemEval-2025 Task 11: Bridging the Gap in Text-Based Emotion Detection (Muhammad et al. 2025)

> Dataset link: https://brighter-dataset.github.io/

The dataset contains sentences labeled with multiple emotions:  
- `anger`  
- `fear`  
- `joy`  
- `sadness`  
- `surprise`

Each sentence can have **more than one emotion** assigned.
---

## 🚀 Getting started

1. Create a virtual environment
   ```
   python -m venv venv
   ```
2. Activate the environmentt
   ```
   venv\Scripts\activate.bat
   ```
3. Install dependencies
   ```
   pip install -r requirements.txt
4. Excecute files 
   ```
   python src/train.py
   ```

   







