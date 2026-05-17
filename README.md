## 📧 Smart Email Spam Classifier

### 💡 About
Ever wondered how Gmail knows which emails are spam?
I built exactly that — an AI-powered Email Spam Classifier 
that automatically detects spam emails using Machine Learning.

No more unwanted emails! 🚫

### 🛠️ Technologies Used

- Python 3.12
- Scikit-learn — AI Model Training
- Pandas — Data Analysis & Processing
- Naive Bayes Algorithm — Spam Classification
- CountVectorizer — Text to Numbers Conversion

### ⚙️ How it Works

1. Email text converted to numerical data
2. Naive Bayes AI model trained on real & spam emails
3. Model predicts Spam or Real with 75% accuracy

### 📊 Model Performance

| Total Emails | Spam | Real | Accuracy |
|-------------|------|------|----------|
| 20 | 10 | 10 | 75% |

### 📤 Output

=== EMAIL SPAM DETECTOR ===

Total Emails: 20

Spam Emails: 10

Real Emails: 10

AI Model Accuracy: 75.0%

=== TESTING EMAILS ===

🚫 SPAM → You won a free iPhone click here to claim

✅ REAL → Please submit your project report by Monday

🚫 SPAM → Earn money fast from home no experience needed

✅ REAL → Your interview is scheduled for tomorrow 10am

🚫 SPAM → Congratulations you have won 10 lakh rupees

### 🚀 How to Run

py -m pip install pandas scikit-learn

py spam_detector.py
