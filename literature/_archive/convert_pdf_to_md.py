"""Simple PDF to Markdown converter for research papers."""

import re
from pathlib import Path
import PyPDF2


def clean_text(text: str) -> str:
    """Clean extracted PDF text."""
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    # Fix common OCR issues
    text = text.replace('ﬁ', 'fi').replace('ﬂ', 'fl')
    # Add line breaks for better readability
    text = re.sub(r'\.(\s+[A-Z])', r'.\n\n\1', text)
    return text.strip()


def pdf_to_markdown(pdf_path: Path, output_dir: Path) -> None:
    """Convert PDF to markdown format."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        
        # Extract metadata
        metadata = reader.metadata
        title = metadata.get('/Title', pdf_path.stem) if metadata else pdf_path.stem
        
        # Start markdown content
        md_content = f"# {title}\n\n"
        
        if metadata:
            if '/Author' in metadata:
                md_content += f"**Authors:** {metadata['/Author']}\n\n"
            if '/Subject' in metadata:
                md_content += f"**Subject:** {metadata['/Subject']}\n\n"
        
        md_content += "---\n\n"
        
        # Extract text from each page
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text.strip():
                md_content += f"## Page {i + 1}\n\n"
                cleaned_text = clean_text(text)
                
                # Try to identify sections
                lines = cleaned_text.split('\n')
                for line in lines:
                    # Check if line might be a header
                    if line.isupper() and len(line) > 3 and len(line) < 100:
                        md_content += f"\n### {line.title()}\n\n"
                    elif re.match(r'^\d+\.?\s+[A-Z]', line):
                        # Numbered section
                        md_content += f"\n### {line}\n\n"
                    else:
                        md_content += line + "\n"
                
                md_content += "\n---\n\n"
        
        # Save markdown file
        output_file = output_dir / f"{pdf_path.stem}.md"
        output_file.write_text(md_content, encoding='utf-8')
        print(f"✅ Converted {pdf_path.name} → {output_file}")


def main():
    """Convert ALFEE and NeuroLM PDFs to markdown."""
    pdfs_dir = Path("/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/literature/pdfs")
    markdown_dir = Path("/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/literature/markdown")
    
    # Convert ALFEE.pdf
    alfee_pdf = pdfs_dir / "ALFEE.pdf"
    if alfee_pdf.exists():
        pdf_to_markdown(alfee_pdf, markdown_dir / "ALFEE")
    else:
        print(f"❌ {alfee_pdf} not found")
    
    # Convert NeuroLM.pdf  
    neurolm_pdf = pdfs_dir / "NeuroLM.pdf"
    if neurolm_pdf.exists():
        pdf_to_markdown(neurolm_pdf, markdown_dir / "NeuroLM")
    else:
        print(f"❌ {neurolm_pdf} not found")


if __name__ == "__main__":
    main()