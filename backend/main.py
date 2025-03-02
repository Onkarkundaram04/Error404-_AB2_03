import os
import random
import re
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from werkzeug.utils import secure_filename

app = Flask(__name__, static_folder="../frontend", static_url_path="")
CORS(app)

# Load dataset and models (once at startup for efficiency)
print("Loading dataset from Hugging Face...")
dataset = load_dataset("epfl-llm/guidelines", split="train")
print(f"Dataset loaded with {len(dataset)} guidelines.")
print("Available column names:", dataset.column_names)

TITLE_COL = "title"
CONTENT_COL = "clean_text"

# Initialize the sentence transformer
print("Initializing sentence transformer...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize the QA model
print("Loading question-answering model...")
qa_pipeline = pipeline(
    "question-answering", model="distilbert-base-cased-distilled-squad"
)


# Define the embedding function
def embed_text(batch):
    combined_texts = [
        title + " " + content[:100]
        for title, content in zip(batch[TITLE_COL], batch[CONTENT_COL])
    ]
    embeddings = embedder.encode(combined_texts, show_progress_bar=False)
    return {"embeddings": embeddings}


print("Embedding guidelines...")
dataset = dataset.map(embed_text, batched=True, batch_size=32)
print("Embeddings generated successfully.")

print("Building FAISS index...")
dataset.add_faiss_index(column="embeddings")
print("FAISS index built successfully.")


def summarize_medical_report(report, qa_pipeline):
    questions = [
        "What is the patient's age?",
        "What is the patient's gender?",
        "What are the patient's current symptoms?",
        "What is the patient's medical history?",
    ]
    answers = []
    for q in questions:
        result = qa_pipeline(question=q, context=report)
        answers.append(result["answer"] if result["score"] > 0.1 else None)

    age = answers[0] if answers[0] else "unknown age"
    gender = answers[1] if answers[1] else "unknown gender"
    symptoms = answers[2] if answers[2] else "unknown symptoms"
    history = answers[3] if answers[3] else "no known medical history"

    summary = f"{age} {gender} with {history}, presenting with {symptoms}"
    return summary


def get_treatment_suggestions(patient_data, k=5):
    print(f"\nProcessing patient data: '{patient_data}'")
    query_embedding = embedder.encode([patient_data])
    scores, retrieved_examples = dataset.get_nearest_examples(
        "embeddings", query_embedding, k=k
    )
    print(f"Retrieved {k} relevant guidelines:")
    for i, (title, score) in enumerate(zip(retrieved_examples[TITLE_COL], scores), 1):
        print(f"{i}. {title} (Similarity Score: {score:.2f})")

    suggestions = []
    for idx, guideline_text in enumerate(retrieved_examples[CONTENT_COL]):
        question = (
            f"What specific medication or procedure is recommended for {patient_data}?"
        )
        result = qa_pipeline(question=question, context=guideline_text)
        treatment_text = result["answer"] if result and "answer" in result else ""
        suggestions.append(
            {
                "guideline": retrieved_examples[TITLE_COL][idx],
                "treatment": treatment_text,
            }
        )
    return suggestions


# Function to extract text from a TXT file
def extract_text_from_txt(txt_path):
    with open(txt_path, "r", encoding="utf-8") as f:
        full_text = f.read()
    print("Extracted TXT Text:\n", full_text)

    extracted_data = {
        "Patient Name": "N/A",
        "Age": "N/A",
        "Gender": "N/A",
        "Medical History": "N/A",
        "Current Symptoms": "N/A",
        "Additional Notes": "N/A",
    }

    patterns = {
        "Patient Name": r"Patient Name[:\-]?\s*(.+)",
        "Age": r"Age[:\-]?\s*(\d+)",
        "Gender": r"(?:Gender|Sex)[:\-]?\s*(\w+)",
        "Medical History": r"Medical History[:\-]?\s*(.+)",
        "Current Symptoms": r"Current Symptoms[:\-]?\s*(.+)",
        "Additional Notes": r"Additional Notes[:\-]?\s*(.+)",
    }

    for field, pattern in patterns.items():
        match = re.search(pattern, full_text, re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            extracted_data[field] = value if value else "N/A"

    print("Extracted Patient Data:", extracted_data)
    return extracted_data


treatment_corrections = {
    "improve symptoms and prolong life": "aspirin",
    "Dyspnea\nAnxiety\nPleuritic chest pain": "nitroglycerin",
    "deep, difficult to localization, and diffuse": "ECG",
    "chronic stable angina": "beta-blockers",
    "anxiety": "oxygen",
}

templates = [
    "Start {treatment} based on {guideline}.",
    "Give {treatment} as per {guideline}.",
    "Consider {treatment} from {guideline}.",
    "{treatment} could work, per {guideline}.",
    "Try {treatment} according to {guideline}.",
]


@app.route("/upload-txt", methods=["POST"])
def upload_txt():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if file and file.filename.lower().endswith(".txt"):
        filename = secure_filename(file.filename)
        temp_dir = os.path.join(os.getcwd(), "data")
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, filename)
        file.save(temp_path)

        try:
            patient_data = extract_text_from_txt(temp_path)
            query_str = (
                f"{patient_data.get('Age', 'unknown')} {patient_data.get('Gender', 'unknown')} "
                f"with {patient_data.get('Medical History', 'no known history')}, presenting with "
                f"{patient_data.get('Current Symptoms', 'no symptoms')}"
            )
            suggestions = get_treatment_suggestions(query_str)
            recommendations = [
                random.choice(templates).format(
                    treatment=treatment_corrections.get(s["treatment"], s["treatment"]),
                    guideline=s["guideline"],
                )
                for s in suggestions
            ]
            response = {"patientData": patient_data, "recommendations": recommendations}
            print("Response Sent:", response)
        except Exception as e:
            response = {"error": f"Processing failed: {str(e)}"}
            return jsonify(response), 500
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

        return jsonify(response)
    return jsonify({"error": "Invalid file format"}), 400


@app.route("/")
def serve_index():
    return app.send_static_file("index.html")


@app.route("/<path:path>")
def serve_static(path):
    return send_from_directory(app.static_folder, path)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
