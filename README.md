# Twitter Sentiment & Hate Speech Detection 🐧🚫

## Watch the Demo 📺

[![YouTube Video](https://img.shields.io/badge/YouTube-Watch%20Demo-red?logo=youtube&logoColor=white&style=for-the-badge)](https://youtu.be/dipnq2uCtr8)

![Image](https://github.com/user-attachments/assets/576d07e5-58bd-44e0-9978-5366eb7f3cba)

## Overview 📚
This project **analyzes Twitter data** to detect **sentiment (positive/negative/neutral) and hate speech** using **machine learning models**. It features a **Streamlit web app** where users can enter text and get predictions.

## Dataset 🗂
- **Source**: Twitter (public datasets on hate speech & sentiment analysis)
- **Features**:
  - Tweet Text
  - Sentiment Labels (Positive, Negative, Neutral)
  - Hate Speech Classification (Hate/Offensive, Neutral)
  
## Machine Learning Models 🧠
The project compares multiple **ML models** for sentiment and hate speech detection:
- **Logistic Regression** 📈
- **Support Vector Machine (SVM)** 🛠️
- **Random Forest** 🌳
- **CatBoost** 🐱
- **Naïve Bayes** 📊
- **Bagging Classifier** 🎭
- **Stacking Model (Best Performing)** 🏆

## Web App Interface 🖥️
The **Streamlit web app** allows users to:
✅ **Enter text or tweet**  
✅ **Get real-time predictions** on sentiment & hate speech  

![Image](https://github.com/user-attachments/assets/0b48a3b0-f59e-42b4-83ee-95b2c6b2d8da)


✅ **See model accuracy comparison**  


![Image](https://github.com/user-attachments/assets/aae9cc7e-56c7-4d46-80aa-97629ca57ab7)


## Files in the Repository 🗁
- `twitter-hate-speech.ipynb` - Jupyter Notebook for data analysis, training models, and accuracy comparison  
- `app.py` - **Streamlit web application** for real-time predictions  
- `stacking_model.pkl` - Saved **best-performing model** for predictions  
- `tfidf_vectorizer.pkl` - **TF-IDF vectorizer** used for text transformation  
- `requirements.txt` - **Dependencies** needed to run the project  

## How to Run 🏃‍♂️
1. **Clone the repository**:  
   ```bash
   git clone https://github.com/yourusername/Twitter-Sentiment-Hate-Speech-Detection.git
   cd Twitter-Sentiment-Hate-Speech-Detection
   ```
2. **Install dependencies**:  
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Streamlit app**:  
   ```bash
   streamlit run app.py
   ```

## Dependencies 📦
- Scikit-learn  
- Streamlit  
- Pandas  
- NumPy  
- CatBoost  
- NLTK  
- Matplotlib  

## Future Improvements 🔮
🔹 Improve model accuracy with **larger datasets**  
🔹 Add **deep learning models (LSTMs, Transformers)**  
🔹 Deploy as a **web API using Flask/FastAPI**  
🔹 Extend support for **multilingual hate speech detection**  

## Author ✍️
- **DataScientist00**
- Kaggle: [Profile](https://www.kaggle.com/codingloading)

---
**⭐ If you found this project helpful, consider giving it a star! ⭐**

