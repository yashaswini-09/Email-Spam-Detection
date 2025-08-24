# Email-Spam-Detection
An end-to-end machine learning project that detects whether an Email or SMS message is Spam or Not Spam.
Built with Python, Streamlit, NLP techniques, and Machine Learning, and ready for deployment on Heroku or Streamlit Cloud
This project provides a simple and interactive Spam Classifier Web App where users can type or paste a message and get instant prediction as Spam or Not Spam. The app uses Natural Language Processing (NLP) for text preprocessing and TF-IDF for feature extraction, combined with a machine learning model trained on a labeled dataset.
End to end code for the email spam classifier project

Features:
✔ Real-time Spam/Not Spam detection
✔ Clean and modern UI with Streamlit
✔ Advanced text preprocessing:
  -Lowercasing
  -Tokenization
  -Stopword Removal
  -Punctuation Removal
  -Stemming using Porter Stemmer

Displays:
✔Transformed text
✔Vectorized input shape

Prediction probabilities:
✔ Pre-trained ML model using TF-IDF + Classifier
✔ Deployment-ready for Heroku or Streamlit Cloud
✔ Includes FAQ section, usage tips, and examples

Dependencies:
-Install all dependencies from requirements.txt:
  streamlit
  nltk
  scikit-learn


Additional:
-nltk datasets: punkt, stopwords (auto-downloaded in app)
-pickle (built-in for model loading)

Technologies & Skills Used:
-Language: Python
-Framework: Streamlit
-NLP Techniques:
  Tokenization
  Stopword Removal
  Stemming
  TF-IDF Vectorization
-Machine Learning:
  Scikit-learn (model training & evaluation)

Deployment:
- Heroku (Procfile & setup.sh)
- Streamlit Sharing (alternative)

Others: Git, GitHub, HTML/CSS customization in Streamlit

How to Run Locally:
-Clone the Repository
  git clone https://github.com/<your-username>/spam_detection.git
  cd spam_detection
-Install Requirements
  pip install -r requirements.txt
-Run the App
  streamlit run app.py


Thankyou for exploring the Project!!

