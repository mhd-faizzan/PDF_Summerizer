import os
import fitz  
import nltk
import streamlit as st
from groq import Groq
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import Table, TableStyle, Paragraph, SimpleDocTemplate, Spacer, PageBreak

# Install necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# Access the Groq API key securely from the secrets file
GROQ_API_KEY = st.secrets["groq"]["api_key"]

# Initialize the Groq client with the API key
client = Groq(api_key=GROQ_API_KEY)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    texts = []
    for page_num in range(min(10, len(doc))):  # Limit to the first 10 pages
        page = doc.load_page(page_num)
        text = page.get_text("text")
        texts.append(text)
    return texts

# Function to summarize text using Groq API
def summarize_text(text, model="llama-3.1-70b-versatile"):
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": text}],
            model=model,
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error during summarization: {e}")
        return "Summary could not be generated."

# Function to extract key terms using Groq API
def extract_key_terms_llama(text, model="llama-3.1-70b-versatile", max_terms=10):
    prompt = (
        "Extract the top {} key terms or phrases from the following text:\n\n"
        "\"\"\"\n"
        "{}\n"
        "\"\"\""
    ).format(max_terms, text)
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model,
        )
        response = chat_completion.choices[0].message.content
        key_terms = [term.strip() for term in response.split(",")]
        return key_terms
    except Exception as e:
        st.error(f"Error during key term extraction: {e}")
        return []

# Function to extract key terms using NLTK
def extract_key_terms_nltk(text, top_n=10):
    words = nltk.word_tokenize(text.lower())
    words = [word for word in words if word.isalpha() and word not in nltk.corpus.stopwords.words('english')]
    tagged_words = nltk.pos_tag(words)
    key_terms = [word for word, pos in tagged_words if pos in ['NN', 'NNS', 'NNP', 'NNPS']]
    fdist = nltk.FreqDist(key_terms)
    common_terms = fdist.most_common(top_n)
    return [term for term, _ in common_terms]

# Function to get word meanings using NLTK
def get_word_meaning(word):
    synsets = nltk.corpus.wordnet.synsets(word)
    if synsets:
        return synsets[0].definition()
    return "No definition found."

# Function to process text and return summary and key terms
def process_text(text, use_llama=False, top_n=10):
    summary = summarize_text(text)
    if use_llama:
        key_terms = extract_key_terms_llama(text, max_terms=top_n)
    else:
        key_terms = extract_key_terms_nltk(text, top_n)
    meanings = {term: get_word_meaning(term) for term in key_terms}
    return summary, meanings

# Function to generate a PDF with summaries and key terms
def generate_pdf(output_path, pages):
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()

    # Custom paragraph style for summaries
    summary_style = ParagraphStyle(
        name="SummaryStyle",
        fontSize=12,
        fontName="Helvetica-Bold",
        leading=18,
        spaceBefore=6,
        spaceAfter=6,
        alignment=1  # Align left
    )

    flowables = []
    for page_content in pages:
        # Add summary paragraph
        summary_paragraph = Paragraph(page_content['summary'], style=summary_style)
        flowables.append(summary_paragraph)

        # Create table for key terms and meanings
        data = [["Term", "Meaning"]]  # Header row
        for term, meaning in page_content['meanings'].items():
            data.append([term, meaning])

        # Define table style
        table_style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), (0.9, 0.9, 0.9)),  # Header background
            ('TEXTCOLOR', (0, 0), (-1, 0), (0, 0, 0)),  # Header text color
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),  # Align text left
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), (1, 1, 1)),  # Alternate row background
            ('GRID', (0, 0), (-1, -1), 1, (0, 0, 0)),  # Grid lines
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),  # Vertical alignment
        ])

        # Define column widths and create table
        col_widths = [2.5*inch, 4.5*inch]  # Adjust column widths as needed
        table = Table(data, colWidths=col_widths)
        table.setStyle(table_style)

        # Add table to flowables
        flowables.append(table)

        # Add page break after each page's content
        flowables.append(PageBreak())

    # Build the PDF
    doc.build(flowables)

# Main function for Streamlit app
def main():
    st.title("PDF Summarizer and Key Term Extractor")

    # PDF File Upload
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        st.write("Processing...")
        pdf_path = "temp_uploaded.pdf"

        # Save the uploaded file temporarily
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Extract text from PDF
        texts = extract_text_from_pdf(pdf_path)

        use_llama = st.checkbox("Use LLaMA for Key Term Extraction", value=True)
        top_n_terms = st.slider("Number of Key Terms to Extract", min_value=1, max_value=20, value=10)

        pages = []
        for idx, text in enumerate(texts[:10], 1):  # Limit to first 10 pages
            summary, meanings = process_text(text, use_llama=use_llama, top_n=top_n_terms)
            page_content = {
                'summary': f"**Page {idx} Summary:**\n{summary}",
                'meanings': meanings
            }
            pages.append(page_content)

        # Output PDF file
        output_pdf_path = "output_summary.pdf"
        generate_pdf(output_pdf_path, pages)

        st.success("PDF generation complete! Download your file below.")
        with open(output_pdf_path, "rb") as f:
            st.download_button("Download PDF", f, file_name=output_pdf_path)

if __name__ == "__main__":
    main()