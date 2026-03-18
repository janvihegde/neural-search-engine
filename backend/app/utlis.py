import pandas as pd
import re

def clean_text(text):
    """
    Basic NLP cleaning: Remove HTML tags, special characters, 
    and extra whitespace to reduce noise for the embedding model.
    """
    if not isinstance(text, str):
        return ""
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove special characters and numbers (optional, but keeps it clean)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase and strip whitespace
    return text.lower().strip()

def load_and_preprocess_data(file_path):
    print(f"--- Loading data from {file_path} ---")
    
    # Load dataset
    df = pd.read_csv(file_path)
    
    # 1. Identify key columns (Kaggle dataset typically uses these)
    # Adjust names if your specific CSV differs
    required_cols = ['Ticket Subject', 'Ticket Description']
    
    # 2. Drop rows with missing critical text
    df = df.dropna(subset=required_cols)
    
    # 3. Create a unified 'combined_text' column for embedding
    # We combine Subject + Description for better context
    df['combined_text'] = df['Ticket Subject'] + " " + df['Ticket Description']
    
    # 4. Apply cleaning
    print("Cleaning text data...")
    df['cleaned_text'] = df['combined_text'].apply(clean_text)
    
    # 5. Keep only what we need to save memory
    processed_df = df[['cleaned_text', 'Ticket Subject', 'Ticket Type', 'Ticket Priority']].copy()
    
    print(f"Successfully processed {len(processed_df)} records.")
    return processed_df

if __name__ == "__main__":
    # Path to your downloaded Kaggle CSV
    DATA_PATH = "customer_support_tickets.csv" 
    
    try:
        data = load_and_preprocess_data(DATA_PATH)
        print(data.head())
        # Save a sample for Phase 2
        data.to_csv("cleaned_tickets.csv", index=False)
    except FileNotFoundError:
        print("Error: Please ensure 'customer_support_tickets.csv' is in the project directory.")