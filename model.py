import pandas as pd
import re
import nltk
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, 
                            recall_score, f1_score, confusion_matrix, 
                            classification_report)

def load_and_prepare_data():
    """Load and prepare the dataset"""
    try:
        df = pd.read_csv("spam.csv", encoding='ISO-8859-1')
        
        if not all(col in df.columns for col in ['v1', 'v2']):
            raise ValueError("CSV file doesn't have expected columns 'v1' and 'v2'")
            
        df = df.rename(columns={'v1': 'target', 'v2': 'message'})
        df = df[['message', 'target']]
        df['target'] = df['target'].map({'ham': 0, 'spam': 1})
        return df
    
    except FileNotFoundError:
        raise FileNotFoundError("'spam.csv' file not found in current directory")
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")

def preprocess_text(text):
    """Clean and preprocess text data"""
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model and return metrics"""
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Ham', 'Spam'], 
                yticklabels=['Ham', 'Spam'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png')
    plt.close()
    
    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'classification_report': report
    }

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Train and evaluate different ML models"""
    models = {
        "Naïve Bayes": Pipeline([
            ('vectorizer', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('classifier', MultinomialNB())
        ]),
        "Logistic Regression": Pipeline([
            ('vectorizer', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('classifier', LogisticRegression(max_iter=1000))
        ]),
        "Random Forest": Pipeline([
            ('vectorizer', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
    }
    
    results = []
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        model_results = evaluate_model(model, X_test, y_test, name)
        results.append(model_results)
        
        # Print metrics
        print(f"\n{name} Performance:")
        print(f"Accuracy: {model_results['accuracy']:.4f}")
        print(f"Precision: {model_results['precision']:.4f}")
        print(f"Recall: {model_results['recall']:.4f}")
        print(f"F1 Score: {model_results['f1_score']:.4f}")
        print("\nClassification Report:")
        print(model_results['classification_report'])
    
    return models, results

def plot_metrics_comparison(results):
    """Visualize model metrics comparison"""
    metrics_df = pd.DataFrame(results)
    
    plt.figure(figsize=(12, 6))
    metrics_df.plot(x='model_name', y=['accuracy', 'precision', 'recall', 'f1_score'], 
                   kind='bar', figsize=(12, 6))
    plt.title("Model Performance Comparison")
    plt.ylabel("Score")
    plt.ylim(0, 1.1)
    plt.xticks(rotation=0)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("metrics_comparison.png")
    plt.show()

def main():
    # Download NLTK resources
    nltk.download('stopwords')
    nltk.download('wordnet')
    
    try:
        # Load and prepare data
        df = load_and_prepare_data()
        
        # Preprocess text
        print("Preprocessing text data...")
        df['message'] = df['message'].apply(preprocess_text)
        
        # Split data
        print("Splitting data into train/test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            df['message'], df['target'], test_size=0.2, random_state=42)
        
        # Train and evaluate models
        print("\nTraining and evaluating models...")
        models, results = train_and_evaluate_models(X_train, X_test, y_train, y_test)
        
        # Save models and results
        with open("spam_models.pkl", "wb") as file:
            pickle.dump({"models": models, "results": results}, file)
        
        # Visualize results
        plot_metrics_comparison(results)
        
        print("\n✅ Process completed successfully!")
        print("Models and evaluation results saved to spam_models.pkl")
        print("Visualizations saved as PNG files")
        
    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")

if __name__ == "__main__":
    main()