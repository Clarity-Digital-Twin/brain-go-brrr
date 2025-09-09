"""Convert SeizureTransformer PDF to Markdown."""

import re
from pathlib import Path
import PyPDF2


def clean_text(text: str) -> str:
    """Clean extracted PDF text."""
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    # Fix common OCR issues
    text = text.replace('ﬁ', 'fi').replace('ﬂ', 'fl')
    text = text.replace('−', '-').replace('–', '-')
    # Add line breaks for better readability
    text = re.sub(r'\.(\s+[A-Z])', r'.\n\n\1', text)
    return text.strip()


def identify_section(line: str) -> tuple:
    """Identify if a line is a section header and return level."""
    # Main sections (often numbered)
    if re.match(r'^\d+\.?\s+[A-Z][a-z]+', line) and len(line) < 100:
        return (2, line)
    # Subsections
    elif re.match(r'^\d+\.\d+\.?\s+', line):
        return (3, line)
    # Abstract, Introduction, etc.
    elif line.strip() in ['Abstract', 'ABSTRACT', 'Introduction', 'INTRODUCTION', 
                          'Methods', 'METHODS', 'Results', 'RESULTS', 
                          'Discussion', 'DISCUSSION', 'Conclusion', 'CONCLUSION',
                          'References', 'REFERENCES', 'Acknowledgments', 'ACKNOWLEDGMENTS']:
        return (2, line.title())
    # All caps headers
    elif line.isupper() and 3 < len(line) < 50 and not line.startswith('IEEE'):
        return (3, line.title())
    return (0, line)


def pdf_to_markdown(pdf_path: Path, output_dir: Path) -> None:
    """Convert PDF to markdown format with better structure."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            
            # Start markdown content
            md_content = "# SeizureTransformer: Scaling U-Net with Transformer for Simultaneous Time-Step Level Seizure Detection from Long EEG Recordings\n\n"
            md_content += "**Authors:** Kerui Wu, Ziyue Zhao, Bülent Yener\n\n"
            md_content += "**Source:** arXiv:2504.00336 (2025)\n\n"
            md_content += "---\n\n"
            
            # Track if we're in references section
            in_references = False
            
            # Extract text from each page
            full_text = ""
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text.strip():
                    full_text += f"\n[PAGE {i+1}]\n" + text
            
            # Clean and process the full text
            cleaned_text = clean_text(full_text)
            
            # Split into lines and process
            lines = cleaned_text.split('\n')
            current_section = ""
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check for page markers
                if line.startswith('[PAGE'):
                    md_content += f"\n\n---\n__{line}__\n\n"
                    continue
                
                # Check if this is References section
                if 'References' in line or 'REFERENCES' in line:
                    in_references = True
                
                # Identify section headers
                level, header = identify_section(line)
                
                if level > 0:
                    md_content += '\n' + '#' * level + f" {header}\n\n"
                    current_section = header
                else:
                    # Format references differently
                    if in_references and line.startswith('['):
                        md_content += f"\n{line}\n"
                    # Check for equations (simple heuristic)
                    elif '=' in line and any(c in line for c in ['α', 'β', 'γ', 'Σ', '∑', '∈']):
                        md_content += f"\n```\n{line}\n```\n"
                    # Check for figure/table captions
                    elif line.startswith(('Figure', 'Table', 'Fig.')):
                        md_content += f"\n**{line}**\n"
                    else:
                        md_content += line + " "
        
        # Save markdown file
        output_file = output_dir / "SeizureTransformer.md"
        output_file.write_text(md_content, encoding='utf-8')
        print(f"✅ Converted {pdf_path.name} → {output_file}")
        
    except Exception as e:
        print(f"❌ Error converting PDF: {e}")
        print("Attempting basic conversion...")
        
        # Fallback to basic conversion
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                md_content = "# SeizureTransformer\n\n"
                
                for i, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text.strip():
                        md_content += f"\n## Page {i + 1}\n\n"
                        md_content += clean_text(text) + "\n\n"
                
                output_file = output_dir / "SeizureTransformer.md"
                output_file.write_text(md_content, encoding='utf-8')
                print(f"✅ Basic conversion completed: {output_file}")
                
        except Exception as e2:
            print(f"❌ Basic conversion also failed: {e2}")


def main():
    """Convert SeizureTransformer PDF to markdown."""
    pdfs_dir = Path("/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/literature/pdfs")
    markdown_dir = Path("/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/literature/markdown/seizure_transformer")
    
    seizure_pdf = pdfs_dir / "SEIZURE_TRANSFORMER.pdf"
    if seizure_pdf.exists():
        pdf_to_markdown(seizure_pdf, markdown_dir)
    else:
        print(f"❌ {seizure_pdf} not found")


if __name__ == "__main__":
    main()