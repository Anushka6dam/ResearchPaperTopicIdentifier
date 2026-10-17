# Research Paper Topic Identification Using NLP and SciBERT

## 📌 Project Overview

Research Paper Topic Identification is a machine learning project that automatically identifies the topic or domain of a research paper based on its title and abstract.

The project uses Natural Language Processing (NLP), traditional machine learning algorithms, and the SciBERT language model to classify research papers into different computer science domains.

## 🎯 Objectives

- Automatically classify research papers into relevant topics.
- Preprocess and clean research paper text.
- Train and compare machine learning classification models.
- Use SciBERT for scientific text classification.
- Provide a simple web interface for predicting research paper topics.

## 🛠️ Technologies Used

- **Programming Language:** Python
- **Natural Language Processing:** NLP
- **Feature Extraction:** TF-IDF
- **Machine Learning:** Linear SVM, Logistic Regression, Naive Bayes
- **Transformer Model:** SciBERT
- **Web Framework:** Flask
- **Frontend:** HTML and CSS
- **Data Processing:** Pandas
- **Visualization:** Matplotlib

## 📚 Dataset

The project uses a dataset containing **15,000 research paper samples** across eight computer science-related categories.

The categories are:

1. Artificial Intelligence
2. Machine Learning
3. Natural Language Processing
4. Computer Vision
5. Cybersecurity
6. Computer Networks
7. Software Engineering
8. Databases

The dataset was prepared using research paper data obtained through Hugging Face resources and processed for classification.

## 🧠 Models Implemented

The following models were trained or evaluated:

- Naive Bayes
- Logistic Regression
- Linear Support Vector Machine (SVM)
- SciBERT

## 📊 Results

The SciBERT model achieved approximately **84.7% accuracy** and an **84.59% F1-score** on the project's evaluation dataset.

The traditional Linear SVM model achieved approximately **83.57% accuracy**.

These results demonstrate the effectiveness of machine learning and transformer-based models for research paper topic classification.

## ⚙️ Project Structure

```text
ResearchPaperTopicIdentifier/
├── app.py
├── run_project.bat
├── README.md
├── .gitignore
├── dataset/
│   └── processed/
├── models/
│   ├── svm_model.pkl
│   └── tfidf_vectorizer.pkl
├── results/
│   ├── model_comparison.csv
│   ├── model_comparison.png
│   ├── confusion_matrix.png
│   ├── topic_distribution.png
│   └── abstract_length_distribution.png
├── src/
│   ├── create_dataset.py
│   ├── preprocessing.py
│   ├── train_model.py
│   ├── model_comparison.py
│   ├── train_scibert.py
│   └── evaluate_scibert.py
├── static/
│   └── style.css
└── templates/
    └── index.html
```

## 🚀 How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/Anushka6dam/ResearchPaperTopicIdentifier.git
```

### 2. Open the project folder

```bash
cd ResearchPaperTopicIdentifier
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment on Windows

```powershell
.\venv\Scripts\Activate.ps1
```

### 5. Install the required packages

If a `requirements.txt` file is available:

```bash
pip install -r requirements.txt
```

If it is not available, install the packages listed in your project's actual dependencies before running the application.

### 6. Run the Flask application

```bash
python app.py
```

### 7. Open the application

Open the local URL displayed in your terminal, usually:

`http://127.0.0.1:5000`

## 💡 Applications

- Research paper classification
- Academic literature organization
- Research domain identification
- Scientific document management
- Assistance with organizing research collections

## 🔮 Future Scope

- Add more research domains.
- Improve classification accuracy.
- Integrate additional scientific language models.
- Support PDF uploads and automatic abstract extraction.
- Deploy the application online.

## 👩‍💻 Author

**Anushka Vijay Damkondwar**

## 📄 License

This project is developed for educational and academic purposes. Add an open-source license if you intend to grant others permission to reuse or distribute the code.
