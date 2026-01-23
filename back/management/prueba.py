import PyPDF2

def extract_pdf_text(pdf_path):
    """Extract text content from a PDF file."""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text
    except FileNotFoundError:
        print(f"Error: File '{pdf_path}' not found.")
        return None
    except Exception as e:
        print(f"Error extracting PDF: {e}")
        return None

# Usage
if __name__ == "__main__":
    pdf_file = r"C:\Users\alvaro\Documents\IATUR\Proyectos\ForecastingPCD\back\management\MTST_PRO_29_09_2025_PUB.PDF"
    content = extract_pdf_text(pdf_file)
    if content:
        print(content)