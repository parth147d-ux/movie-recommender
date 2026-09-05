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
## 💡 Key Highlights

- Built end-to-end ML pipeline from scratch  
- Implemented content-based recommendation system  
- Used cosine similarity for accurate results  
- Added fuzzy matching for handling user input errors  
- Designed modular and scalable project structure  

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
## 📸 Demo

Example:

Input: Avengers  

Output:
- Iron Man  
- Captain America: Civil War  
- Avengers: Age of Ultron  

## 🖥️ UI Preview

<p align="center">
  <img src="https://github.com/user-attachments/assets/581c21f7-5adc-4ca1-ba02-b38c54e85710" width="800"/>
</p>



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
