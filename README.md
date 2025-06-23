# 💼 Job Market Cheat Codes (Experimental Edition)

> A machine learning pipeline for analyzing job listings — built entirely on synthetic data, fine-tuned with GPT — to predict salaries, classify job roles, and cluster careers like a data wizard 🎯🧠

📄 **[Read the full paper on arXiv](https://arxiv.org/abs/2506.15879)**  
📥 Or [download the PDF directly from this repo](./Research Paper.pdf)

[![arXiv](https://img.shields.io/badge/arXiv-2506.15879-b31b1b.svg)](https://arxiv.org/abs/2506.15879)
---

## 🧪 Project Overview

This repository contains an end-to-end machine learning framework that demonstrates how we can extract insights from job listings — even when using a **fully synthetic dataset**. 

It was designed as an experimental sandbox to prototype models and validate a scalable workflow for:

- 💰 Salary Prediction (Regression)
- 🧑‍💼 Job Title Classification
- 🔗 Job Role Clustering
- ✍️ Feature Engineering + NLP (TF-IDF, SBERT)
- 📊 Data Visualization & Interpretability

All work was conducted on a fabricated dataset of ~1.6M samples, generated via the Python Faker library and structured with the help of ChatGPT.

---

## 🧠 Why Synthetic?

Real job listing datasets are often:
- Incomplete
- Inaccessible
- Biased
- Messy as hell

We used synthetic data to:
- Develop and test modeling pipelines in a clean, controlled environment
- Prevent privacy issues or ethical concerns
- Focus on **workflow**, not real-world generalization

📌 **Disclaimer:**  
> This dataset is **NOT suitable for real-world applications**. This project is for experimentation and educational purposes only.

---

## 📂 Project Structure

```bash
Job-Market-Cheat-Codes/
│
├── Paper.pdf                # Full research paper
├── Notebooks/
│   ├── 1-EDA.ipynb                   # Exploratory Data Analysis
│   ├── 2-Preprocessing.ipynb         # Heavy feature engineering + text masking + embeddings
│   ├── 3.1-Regression.ipynb          # Salary prediction models (Ridge, KNN, SVR)
│   ├── 3.2-Classification.ipynb      # Job title prediction (LogReg, KNN)
│   └── 3.3-Clustering.ipynb          # K-Means clustering (TF-IDF & SBERT)
├── Python Executable Scripts/
│   ├── 1-EDA.py                   
│   ├── 2-Preprocessing.py         
│   ├── 3.1-Regression.py        
│   ├── 3.2-Classification.py   
│   └── 3.3-Clustering.py
├── links.txt                         # Links to the original dataset, our notebooks on kaggle and the research paper on overleaf
├── LICENCE                           # MIT Licence
└── README.md                         # This file!
```

## 🔧 Tech Stack
- 🐍 Python (Jupyter Notebooks)

- RAPIDS cuML (GPU-accelerated ML)

- scikit-learn

- SentenceTransformers (SBERT)

- Faker (for synthetic data gen)

- TF-IDF + PCA + KMeans

- Matplotlib / Seaborn

## 📈 Key Results
Task / Best Model / Score
- Salary Regression /	Ridge Regression /	RMSE ≈ 0 (synthetic-boosted)
- Job Classification /	Logistic Regression /	F1 ≈ 0.98
- Job Clustering /	KMeans + TF-IDF (K=40) /	DB Score ≈ 2.37

🧨 **Note:** These performance scores are exaggerated due to the synthetic, well-structured nature of the data. Real-world datasets will have more complexity and noise.

## 🧑‍💻 Authors
- Abdel Rahman Alsheyab – arahmadalsheyab22@cit.just.edu.jo

- Mohammad Alkasawneh – myalkhasawneh22@cit.just.edu.jo

- Nidal Shahin – nkhameedshahin22@cit.just.edu.jo

## 📝 Citation & Credits
If you're using this project for reference or learning, a simple shoutout would be appreciated 🙌

Special thanks to:

- Ravindra Rana’s Synthetic Job Dataset on Kaggle:
    - (https://www.kaggle.com/datasets/ravindrasinghrana/job-description-dataset)

## 🚀 Future Work
This project is just the blueprint. Next steps:

- ✅ Replace synthetic data with a real-world dataset

- ✅ Fine-tune models on messier, biased, more challenging data

- ✅ Expand the clustering analysis into career path recommendation systems

- ✅ Create a web-based dashboard (Power BI / Streamlit)

## 💡 License
This repository is licensed under the MIT License — free to use, modify, and share, but not to pretend you're building a real-world job prediction engine with fake data!
