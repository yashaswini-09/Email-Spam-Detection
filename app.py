import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Page configuration
st.set_page_config(page_title="Email/SMS Spam Classifier 📩", page_icon="📬", layout="centered")

# Custom CSS styles
st.markdown("""
<style>
    .stTextArea > div > textarea {
        font-size: 16px;
        min-height: 120px;
        border-radius: 8px;
        padding: 12px;
        border: 2px solid #ccc;
        transition: border-color 0.3s ease-in-out;
    }
    .stTextArea > div > textarea:focus {
        border-color: #0072C6;
        outline: none;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        font-size: 20px;
        font-weight: 600;
        color: white;
    }
    .spam {
        background-color: #d9534f;
    }
    .notspam {
        background-color: #5cb85c;
    }
    .explanation {
        font-size: 14px;
        color: #444;
        margin-top: 10px;
        margin-bottom: 30px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar content without emojis, with tips
with st.sidebar:
    st.header("How to Use")
    st.markdown("""
    1. **Type or paste** your message in the box.  
    2. Click **Predict**.  
    3. See if your message is classified as **Spam** or **Not Spam**.  
    4. Use example messages below for quick testing.
    """)
    st.markdown("---")

    st.header("Examples")
    if st.button("Example Spam"):
        st.session_state['input_sms'] = "WIN a FREE iPhone now!!! Click to claim your prize."
    if st.button("Example Not Spam"):
        st.session_state['input_sms'] = "Hey, are we still meeting for lunch tomorrow?"
    st.markdown("---")

    st.header("Tips for Accurate Spam Detection")
    st.markdown("""
    - Keep your message clear and concise.  
    - Avoid slang or excessive emojis.  
    - Use complete sentences for context.  
    - Rephrase if unsure about prediction.  
    - Be cautious with suspicious links or offers.
    """)

# Main page headline/title
st.title("📩 Email/SMS Spam Classifier")
st.write("Type your message below and click Predict to check if it’s spam or not.")

# Input box with session state for example buttons
if 'input_sms' not in st.session_state:
    st.session_state['input_sms'] = ""

input_sms = st.text_area("Enter your message here:", value=st.session_state['input_sms'], key="input_sms")

# Prediction button and logic
if st.button('Predict'):
    if not input_sms.strip():
        st.warning("Please enter a message to classify.")
    else:
        # 1. Preprocess
        transformed_sms = transform_text(input_sms)
        st.markdown(f"**🔄 Transformed Text:** `{transformed_sms}`")

        # 2. Vectorize
        vector_input = tfidf.transform([transformed_sms])
        st.markdown(f"**🔢 Vectorized Input Shape:** {vector_input.shape}")

        # 3. Predict
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(vector_input)[0]
            st.markdown(f"**📊 Prediction Probabilities:** Spam: {proba[1]:.2f} | Not Spam: {proba[0]:.2f}")

        result = model.predict(vector_input)[0]
        st.markdown(f"**🎯 Raw Prediction Result:** {result}")

        # 4. Show result with one emoji only
        if result == 1:
            st.markdown(
                '<div class="result-box spam">🚨 This message is classified as <strong>Spam</strong>.</div>',
                unsafe_allow_html=True
            )
            st.markdown("""
                <div class="explanation">
                Spam messages often include unwanted ads, scams, or phishing attempts. Be cautious!
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="result-box notspam">✅ This message is classified as <strong>Not Spam</strong>.</div>',
                unsafe_allow_html=True
            )
            st.markdown("""
                <div class="explanation">
                This message seems safe and not spammy. You can open or reply.
                </div>
            """, unsafe_allow_html=True)

# FAQ expander
with st.expander("❓ Frequently Asked Questions"):
    st.markdown("""
    **Q:** What is spam?  
    **A:** Unsolicited messages, often advertising or scams.

    **Q:** Can the model be 100% accurate?  
    **A:** No, but it's trained to give reliable predictions.

    **Q:** How is the text processed?  
    **A:** Cleaned, tokenized, stopwords removed, and stemmed before prediction.

    **Q:** What to do if I get spam?  
    **A:** Avoid clicking suspicious links and report/block sender.
    """)


