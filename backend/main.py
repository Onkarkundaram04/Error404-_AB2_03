import os
import json
import uuid
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import google.generativeai as genai
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
import markdown
import re
import traceback

# Configuration
GEMINI_API_KEY = "AIzaSyDGpNmvskXEAeOH6hG_BtT8GR043tMREYk"
genai.configure(api_key=GEMINI_API_KEY)

# Initialize Flask app
app = Flask(__name__, static_folder="../frontend", static_url_path="")
CORS(app)

# RAG Model Initialization
print("🚀 Initializing RAG System...")

# Load medical guidelines dataset
print("📂 Loading dataset...")
dataset = load_dataset("epfl-llm/guidelines", split="train")
TITLE_COL = "title"
CONTENT_COL = "clean_text"

# Initialize models
print("🤖 Loading AI models...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
qa_pipeline = pipeline(
    "question-answering", model="distilbert-base-cased-distilled-squad"
)

# Build FAISS index
print("🔍 Building FAISS index...")


def embed_text(batch):
    combined_texts = [
        f"{title} {content[:200]}"
        for title, content in zip(batch[TITLE_COL], batch[CONTENT_COL])
    ]
    return {"embeddings": embedder.encode(combined_texts, show_progress_bar=False)}


dataset = dataset.map(embed_text, batched=True, batch_size=32)
dataset.add_faiss_index(column="embeddings")


# Processing Functions
def format_response(text):
    """Convert Markdown text to HTML for proper frontend display."""
    return markdown.markdown(text)


def extract_patient_info(report):
    """Extract patient information using QA pipeline."""
    questions = [
        "What is the patient's name?",
        "What is the patient's age?",
        "What is the patient's gender?",
        "What are the current symptoms?",
        "What is the medical history?",
    ]

    answers = {}
    for q in questions:
        result = qa_pipeline(question=q, context=report)
        if q == "What is the patient's name?":
            answers["name"] = result["answer"] if result["score"] > 0.1 else "Unknown"
        elif q == "What is the patient's age?":
            answers["age"] = result["answer"] if result["score"] > 0.1 else "Unknown"
        elif q == "What is the patient's gender?":
            answers["gender"] = result["answer"] if result["score"] > 0.1 else "Unknown"
        elif q == "What are the current symptoms?":
            answers["symptoms"] = (
                result["answer"] if result["score"] > 0.1 else "Not specified"
            )
        elif q == "What is the medical history?":
            answers["history"] = (
                result["answer"] if result["score"] > 0.1 else "Not specified"
            )
    return answers


def summarize_report(report):
    """Generate a clinical summary using QA and Gemini model."""
    patient_info = extract_patient_info(report)

    model = genai.GenerativeModel("gemini-2.0-flash")
    prompt = f"""Create clinical summary from:
    - Name: {patient_info['name']}
    - Age: {patient_info['age']}
    - Gender: {patient_info['gender']}
    - Symptoms: {patient_info['symptoms']}
    - History: {patient_info['history']}
    
    Format as: "[{patient_info['name']}] is a [{patient_info['age']}]-year-old [{patient_info['gender']}] with [{patient_info['history']}], presenting with [{patient_info['symptoms']}]"
    Add relevant medical context."""

    summary_text = model.generate_content(prompt).text.strip()
    return format_response(summary_text), patient_info


def get_reference_url(source_name):
    """Convert source names to actual URLs."""
    url_mappings = {
        "CDC": "https://www.cdc.gov/guidelines/",
        "WHO": "https://www.who.int/publications/guidelines/",
        "NIH": "https://www.nih.gov/health-information/guidelines/",
        "ADA": "https://diabetes.org/clinical-guidance/",
        "AHA": "https://professional.heart.org/guidelines/",
        "AAFP": "https://www.aafp.org/clinical-recommendations/",
        "ACR": "https://www.rheumatology.org/practice-quality/clinical-support/clinical-practice-guidelines/",
        "NICE": "https://www.nice.org.uk/guidance/",
        "ACP": "https://www.acponline.org/clinical-information/guidelines/",
        "JAMA": "https://jamanetwork.com/journals/jama/clinical-guidelines/",
        "NEJM": "https://www.nejm.org/medical-guidelines",
        "Lancet": "https://www.thelancet.com/clinical/diseases",
        "BMJ": "https://www.bmj.com/uk/clinicalguidelines",
        "Mayo Clinic": "https://www.mayoclinic.org/medical-professionals/clinical-updates/",
    }

    for key, url in url_mappings.items():
        if key.lower() in source_name.lower():
            return {"name": source_name, "url": url}

    url_pattern = r"https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+"
    urls = re.findall(url_pattern, source_name)
    if urls:
        return {"name": source_name, "url": urls[0]}

    search_url = f"https://www.google.com/search?q={source_name.replace(' ', '+')}+medical+guidelines"
    return {"name": source_name, "url": search_url}


def rag_retrieval(query, k=3):
    """Retrieve relevant guidelines using FAISS."""
    query_embedding = embedder.encode([query])
    scores, examples = dataset.get_nearest_examples("embeddings", query_embedding, k=k)
    return [
        {
            "title": title,
            "content": content[:1000],
            "source": examples.get("source", ["N/A"] * len(examples[TITLE_COL]))[i],
            "score": float(score),
        }
        for i, (title, content, score) in enumerate(
            zip(
                examples[TITLE_COL],
                examples[CONTENT_COL],
                scores,
            )
        )
    ]


def filter_results_by_keywords(results, keywords):
    """Filter search results by keywords."""
    if not keywords:
        return results

    filtered_results = []
    keywords = [k.lower() for k in keywords]

    for result in results:
        title_lower = result["title"].lower()
        content_lower = result.get("content", "").lower()
        snippet_lower = result.get("snippet", "").lower()

        if any(
            k in title_lower or k in content_lower or k in snippet_lower
            for k in keywords
        ):
            filtered_results.append(result)

    return filtered_results


def search_research_papers(query, max_results=5):
    """Search for highly relevant recent research papers using Gemini Flash."""
    model = genai.GenerativeModel("gemini-2.0-flash")

    # Create a more structured prompt that explicitly requests properly formatted data
    prompt = f"""Generate a JSON array of {max_results} highly relevant and recent (2023-2024) medical research papers related to "{query}".
    
    For each paper include EXACTLY these fields:
    - "title": Full title of the paper (must be specific, not generic)
    - "authors": String with 2-3 author names separated by commas (no arrays)
    - "journal": Specific journal name (e.g., "New England Journal of Medicine", "The Lancet", "JAMA")
    - "date": Publication date in YYYY-MM-DD format (must be between 2023-01-01 and 2024-10-31)
    - "doi": A valid DOI or direct URL to the paper (must start with "https://" and point to the specific paper)
    - "summary": A concise 2-3 sentence summary of the key findings
    
    IMPORTANT REQUIREMENTS:
    1. Each paper MUST be real and published in a reputable medical journal
    2. Every field MUST contain valid data (no placeholders, no "None", no empty values)
    3. The "doi" field MUST contain a working URL that leads directly to the paper
    4. The "authors" field MUST be a plain string, NOT an array
    5. Focus on high-impact research from major medical journals
    
    Return only valid JSON array of objects with no additional text.
    """

    try:
        # Set safety settings to allow for more comprehensive information
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]

        # Generate response with increased temperature for more varied results
        response = model.generate_content(
            prompt,
            generation_config={"temperature": 0.2, "max_output_tokens": 2048},
            safety_settings=safety_settings,
        )

        response_text = response.text.strip()

        # Clean response to ensure it's valid JSON
        # Remove any markdown code block indicators or extra text
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()

        # Parse JSON
        papers = json.loads(response_text)

        # Validate and fix each paper entry
        valid_papers = []
        for paper in papers:
            # Skip papers with missing required fields
            if not all(
                key in paper and paper[key]
                for key in ["title", "authors", "journal", "date", "doi", "summary"]
            ):
                continue

            # Convert authors array to string if needed
            if isinstance(paper["authors"], list):
                paper["authors"] = ", ".join(paper["authors"])

            # Ensure DOI is a proper URL
            if not paper["doi"].startswith("http"):
                paper["doi"] = f"https://doi.org/{paper['doi']}"

            # Add to valid papers
            valid_papers.append(paper)

        # If no valid papers were found, create a fallback entry
        if not valid_papers:
            valid_papers.append(
                {
                    "title": f"Recent research on {query}",
                    "authors": "Medical research community",
                    "journal": "PubMed Central",
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "doi": f"https://pubmed.ncbi.nlm.nih.gov/?term={query.replace(' ', '+')}",
                    "summary": f"Search for recent research papers on {query}. Click to view PubMed results.",
                }
            )

        # Sort by date (most recent first)
        valid_papers.sort(key=lambda x: x.get("date", ""), reverse=True)

        return valid_papers[:max_results]
    except Exception as e:
        print(f"Research search error: {str(e)}")
        # Return a placeholder entry
        return [
            {
                "title": f"Search for '{query}' papers",
                "authors": "Medical Database",
                "journal": "PubMed Central",
                "date": datetime.now().strftime("%Y-%m-%d"),
                "doi": f"https://pubmed.ncbi.nlm.nih.gov/?term={query.replace(' ', '+')}",
                "summary": f"Click to search PubMed for papers on {query}.",
            }
        ]


def generate_recommendations(report):
    """Generate treatment recommendations with RAG context."""
    guidelines = rag_retrieval(report)
    context = "Relevant Clinical Guidelines:\n" + "\n".join(
        [f"• {g['title']}: {g['content']} [Source: {g['source']}]" for g in guidelines]
    )

    model = genai.GenerativeModel("gemini-2.0-flash")
    prompt = f"""Generate treatment recommendations using these guidelines:
    {context}
    
    Patient Presentation:
    {report}
    
    Format with:
    - Bold section headers
    - Clear bullet points
    - Evidence markers [Guideline #]
    - Risk-benefit analysis
    - Include references to the sources provided where applicable
    """
    recommendations = model.generate_content(prompt).text.strip()

    reference_objects = []
    for g in guidelines:
        if g["source"] != "N/A":
            reference_objects.append(get_reference_url(g["source"]))

    return format_response(recommendations), reference_objects


def generate_risk_assessment(summary):
    """Generate risk assessment using the summary."""
    model = genai.GenerativeModel("gemini-2.0-flash")
    prompt = f"""Analyze clinical risk:
    {summary}
    
    Output format:
    Risk Score: 0-100
    Alert Level: 🔴 High/🟡 Medium/🟢 Low
    Key Risk Factors: bullet points
    Recommended Actions: bullet points"""
    return format_response(model.generate_content(prompt).text.strip())


# Flask Endpoints
@app.route("/upload-txt", methods=["POST"])
def handle_upload():
    """Handle text file upload and return processed data."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file or not file.filename.endswith(".txt"):
        return jsonify({"error": "Invalid file, must be a .txt file"}), 400

    try:
        content = file.read().decode("utf-8")
        if not content.strip():
            return jsonify({"error": "File is empty"}), 400

        summary, patient_info = summarize_report(content)
        recommendations, references = generate_recommendations(content)
        risk_assessment = generate_risk_assessment(summary)

        return jsonify(
            {
                "session_id": str(uuid.uuid4()),
                "timestamp": datetime.now().isoformat(),
                "summary": summary,
                "patient_info": patient_info,
                "recommendations": recommendations,
                "risk_assessment": risk_assessment,
                "references": references,
            }
        )
    except Exception as e:
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500


@app.route("/search", methods=["GET"])
def search_knowledge():
    """Unified search endpoint for guidelines and research papers."""
    query = request.args.get("q", "")
    keywords = request.args.getlist("keywords")
    research_only = request.args.get("research_only", "false").lower() == "true"

    if not query.strip():
        return jsonify({"error": "Empty query"}), 400

    try:
        combined_results = []

        # Retrieve guidelines unless research_only is set
        if not research_only:
            guideline_results = rag_retrieval(query, k=5)

            # Add guidelines to results
            for i, result in enumerate(guideline_results):
                source_info = get_reference_url(result["source"])
                combined_results.append(
                    {
                        "type": "guideline",
                        "id": f"g{i+1}",
                        "title": result["title"],
                        "snippet": (
                            result["content"][:200] + "..."
                            if len(result["content"]) > 200
                            else result["content"]
                        ),
                        "relevance": int(result["score"] * 100),
                        "source": source_info["name"],
                        "url": source_info["url"],
                    }
                )

        # Always retrieve research papers
        research_results = search_research_papers(query)

        # Add research papers to results with improved formatting
        for i, paper in enumerate(research_results):
            # Format authors safely - handling different possible formats
            authors = paper.get("authors", [])
            if isinstance(authors, list):
                authors_display = ", ".join(authors[:3])  # Take first 3 authors
            elif isinstance(authors, str):
                authors_display = authors
            else:
                authors_display = "Unknown authors"

            if len(authors_display) > 50:
                authors_display = authors_display[:47] + "..."

            # Format summary
            summary = paper.get("summary", "No summary available")
            snippet = summary[:240] + "..." if len(summary) > 240 else summary

            # Calculate days since publication for relevance sorting
            try:
                pub_date = datetime.strptime(
                    paper.get("date", "2023-01-01"), "%Y-%m-%d"
                )
                days_since_pub = (datetime.now() - pub_date).days
                # Newer papers get higher relevance (max 100)
                date_relevance = max(0, 100 - min(days_since_pub // 30 * 5, 50))
            except Exception as date_error:
                print(f"Date parsing error: {str(date_error)}")
                date_relevance = 50

            # Ensure URL is valid
            paper_url = paper.get("doi", "")
            if not paper_url or paper_url == "None" or paper_url is None:
                paper_url = (
                    f"https://pubmed.ncbi.nlm.nih.gov/?term={query.replace(' ', '+')}"
                )

            combined_results.append(
                {
                    "type": "research",
                    "id": f"r{i+1}",
                    "title": paper.get("title", "Untitled Paper"),
                    "snippet": snippet,
                    "authors": authors_display,
                    "journal": paper.get("journal", "Unknown Journal"),
                    "date": paper.get("date", "Unknown Date"),
                    "url": paper_url,
                    "relevance": date_relevance,
                    "source": paper.get("journal", "Research Paper"),
                }
            )

        # Apply keyword filtering
        if keywords:
            combined_results = filter_results_by_keywords(combined_results, keywords)

        # Sort by relevance
        combined_results.sort(key=lambda x: x.get("relevance", 0), reverse=True)

        return jsonify(
            {
                "query": query,
                "results": combined_results,
                "timestamp": datetime.now().isoformat(),
                "total_count": len(combined_results),
                "research_count": sum(
                    1 for r in combined_results if r["type"] == "research"
                ),
                "guideline_count": sum(
                    1 for r in combined_results if r["type"] == "guideline"
                ),
            }
        )
    except Exception as e:
        print(f"Search error: {str(e)}")
        traceback_info = traceback.format_exc()
        print(f"Traceback: {traceback_info}")
        return (
            jsonify(
                {
                    "error": f"Search failed: {str(e)}",
                    "query": query,
                    "results": [],
                    "timestamp": datetime.now().isoformat(),
                }
            ),
            500,
        )


@app.route("/")
def serve_index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/<path:path>")
def serve_static(path):
    return send_from_directory(app.static_folder, path)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
