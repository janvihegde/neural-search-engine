import pandas as pd
import os
df = pd.read_csv("data/raw/customer_support_tickets.csv", nrows=0)
print(df.columns.tolist())

# 1. Setup Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DATA = os.path.join(BASE_DIR, "data", "raw", "customer_support_tickets.csv")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
OUTPUT_FILE = os.path.join(PROCESSED_DIR, "cleaned_tickets.csv")

def prepare():
    if not os.path.exists(RAW_DATA):
        print(f"❌ Could not find raw data at {RAW_DATA}")
        return

    df = pd.read_csv(RAW_DATA)
    print("Actual columns found:", df.columns.tolist())

    # Map your Instruction Dataset columns to our Search Engine Schema
    # 'instruction' -> The query-able header
    # 'response'    -> The content we show the user
    subj_col = 'instruction'
    desc_col = 'response'

    print(f"Mapping columns: '{subj_col}' as Subject and '{desc_col}' as Description")
    
    # 1. Drop rows with missing text
    df = df.dropna(subset=[subj_col, desc_col])
    
    # 2. Create 'cleaned_text' by combining intent, category, and instruction
    # This gives the model the best chance at understanding the "meaning"
    df['cleaned_text'] = (
        df['intent'].astype(str) + " " + 
        df['category'].astype(str) + " " + 
        df['instruction'].astype(str)
    )
    
    # 3. Create the standardized output dataframe
    output_df = pd.DataFrame({
        'Ticket Subject': df[subj_col],
        'Ticket Description': df[desc_col],
        'Category': df['category'],
        'cleaned_text': df['cleaned_text']
    })

    # Ensure processed directory exists
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    # 4. Save to CSV
    output_df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Success! Created {OUTPUT_FILE} with {len(output_df)} rows.")
if __name__ == "__main__":
    prepare()