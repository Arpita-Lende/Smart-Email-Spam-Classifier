import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- DATASET ---
data = {
    'email': [
        # Spam emails
        "Win a free iPhone now click here",
        "You have won 1 crore lottery claim now",
        "Free money waiting for you click link",
        "Congratulations you won a prize collect now",
        "Get rich quick scheme earn money fast",
        "Buy cheap medicines online discount offer",
        "You are selected for free vacation claim",
        "Urgent your account needs verification click",
        "Make money from home easy job offer",
        "Free gift card winner click to claim",
        
        # Real emails
        "Meeting scheduled for tomorrow at 10am",
        "Please review the project report attached",
        "Your order has been delivered successfully",
        "Team lunch is planned for Friday",
        "Can you send me the assignment notes",
        "Interview call letter from HR department",
        "Your college fees receipt is attached",
        "Project submission deadline is next Monday",
        "Please attend the seminar on Data Analysis",
        "Your internship certificate is ready collect",
    ],
    'label': [
        'spam','spam','spam','spam','spam',
        'spam','spam','spam','spam','spam',
        'ham','ham','ham','ham','ham',
        'ham','ham','ham','ham','ham'
    ]
}

df = pd.DataFrame(data)

# --- DATA ANALYSIS ---
print("=== EMAIL SPAM DETECTOR ===")
print(f"\nTotal Emails: {len(df)}")
print(f"Spam Emails: {len(df[df['label']=='spam'])}")
print(f"Real Emails: {len(df[df['label']=='ham'])}")

# --- AI MODEL ---
# Convert text to numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['email'])
y = df['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train AI model
model = MultinomialNB()
model.fit(X_train, y_train)

# Check accuracy
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"\nAI Model Accuracy: {accuracy*100:.1f}%")

# --- TEST YOUR OWN EMAILS ---
print("\n=== TESTING EMAILS ===")
test_emails = [
    "You won a free iPhone click here to claim",
    "Please submit your project report by Monday",
    "Earn money fast from home no experience needed",
    "Your interview is scheduled for tomorrow 10am",
    "Congratulations you have won 10 lakh rupees",
]

for email in test_emails:
    email_vectorized = vectorizer.transform([email])
    result = model.predict(email_vectorized)[0]
    if result == 'spam':
        print(f"🚫 SPAM    → {email}")
    else:
        print(f"✅ REAL    → {email}")

print("\nDone!")