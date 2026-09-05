# 🎬 Movie Recommendation System

A content-based movie recommendation system built using Python and machine learning techniques.  
The system suggests movies similar to a given input by analyzing movie metadata such as genres, cast, keywords, and overview.

---

## 🚀 Features

- Recommend top 5 similar movies
- Content-based filtering using metadata
- Handles partial inputs and typos using fuzzy matching
- Clean modular code structure
- Optional Streamlit UI for interactive usage

---

## 🧠 How It Works

1. Movie datasets are loaded and merged  
2. Important features (genres, keywords, cast, crew, overview) are combined into a single column  
3. Text data is converted into numerical vectors using **CountVectorizer**  
4. **Cosine similarity** is used to measure similarity between movies  
5. Based on similarity scores, the top 5 closest movies are recommended  

---

## 🛠️ Tech Stack

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Difflib (for fuzzy matching)  
- Streamlit (for UI)  

---

## 📂 Project Structure

movie-recommender/
│
├── data/
├── src/
│ ├── preprocess.py
│ ├── model.py
│ └── recommend.py
│
├── app/
│ └── app.py
│
├── main.py
├── requirements.txt
└── README.md

---

## ▶️ How to Run

### 1. Clone the repository
git clone https://github.com/parth147d-ux/movie-recommender.git
cd movie-recommender

### 2. Install dependencies
pip install -r requirements.txt

### 3. Run the project
python main.py

---

## 📊 Dataset

- TMDB 5000 Movies Dataset (Kaggle)

---

## ✨ Future Improvements

- Add web UI (Streamlit)
- Show movie posters
- Deploy online

---

## 👤 Author

**Parth Dwivedi**  
📧 parth147d@gmail.com  
🔗 GitHub: https://github.com/parth147d-ux