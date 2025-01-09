import os
import re
import pandas as pd
from transformers import pipeline

# Initialize NLP models
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)

# Load keywords from a CSV file
def load_keywords_from_csv(csv_file_path):
    df = pd.read_csv(csv_file_path)
    return {row["Category"]: row["Keywords"].split(", ") for _, row in df.iterrows()}

# Summarize the text
def summarize_text(text):
    return summarizer(text, max_length=130, min_length=30, do_sample=False)[0]['summary_text']

# Extract context-based descriptions
def extract_keywords_with_context(text, keywords):
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    matched_keywords_with_context = []
    for keyword in keywords:
        for sentence in sentences:
            if re.search(rf"\b{re.escape(keyword)}\b", sentence, re.IGNORECASE):
                match = re.search(rf"({keyword}.*?)([.!?]|$)", sentence, re.IGNORECASE)
                if match:
                    description = match.group(1).strip()
                    matched_keywords_with_context.append(description)
                break
    return "; ".join(matched_keywords_with_context) if matched_keywords_with_context else "Not specified"

# Extract budget and timeline using regex
def extract_budget_and_timeline(text):
    budget_match = re.search(r"€[0-9,]+", text)
    budget = budget_match.group(0) if budget_match else "Not specified"
    timeline_match = re.search(r"\b(\d+\s*(months?|Monate?n?))", text, re.IGNORECASE)
    timeline = timeline_match.group(0) if timeline_match else "Not specified"
    return budget, timeline

# Process the input text and generate a report
def process_text(text, categories_keywords):
    summary = summarize_text(text)
    extracted_data = {}

    # Extract contextual keywords for categories like Rooms and Special Features
    for category, keywords in categories_keywords.items():
        extracted_data[category] = extract_keywords_with_context(text, keywords)

    # Extract budget and timeline
    budget, timeline = extract_budget_and_timeline(text)
    extracted_data["Budget"] = budget
    extracted_data["Timeline"] = timeline

    # Additional Notes and Project Description
    extracted_data["Project Description"] = summary
    extracted_data["Additional Notes"] = "Details not specified"  # Placeholder for now

    return extracted_data

# Consolidate data into a DataFrame
def consolidate_to_report(data):
    report_row = {
        "Project Description": data.get("Project Description", "Not specified"),
        "Rooms": data.get("Rooms", "Not specified"),
        "Special Features": data.get("Special Features", "Not specified"),
        "Design Style": data.get("Design Style", "Not specified"),
        "Materials": data.get("Materials", "Not specified"),
        "Budget": data.get("Budget", "Not specified"),
        "Timeline": data.get("Timeline", "Not specified"),
        "Additional Notes": data.get("Additional Notes", "Not specified"),
    }
    return pd.DataFrame([report_row])

# Process all files in a folder and generate individual reports
def process_and_save_reports(conversations_folder, categories_keywords, output_folder="reports"):
    os.makedirs(output_folder, exist_ok=True)  # Create folder for reports if it doesn't exist

    # Find all text files in the folder
    conversation_files = [f for f in os.listdir(conversations_folder) if f.endswith('.txt')]

    for file_name in conversation_files:
        file_path = os.path.join(conversations_folder, file_name)
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
        
        # Process the text
        result = process_text(text, categories_keywords)
        df_report = consolidate_to_report(result)

        # Save the report with a meaningful file name
        output_file = os.path.join(output_folder, f"report_{os.path.splitext(file_name)[0]}.csv")
        df_report.to_csv(output_file, index=False)
        print(f"Report saved to {output_file}")

# Main workflow
if __name__ == "__main__":
    # Path to the folder with conversation texts
    conversations_folder = "conversations"

    # Path to the CSV file containing categories and keywords
    csv_file_path = "keywords_by_category.csv"

    # Load keywords
    categories_keywords = load_keywords_from_csv(csv_file_path)

    # Process all files and save reports
    process_and_save_reports(conversations_folder, categories_keywords)
