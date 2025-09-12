"""Convert MNE PDFs to Markdown using PyPDF2."""

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


def pdf_to_markdown(pdf_path: Path, output_dir: Path) -> None:
    """Convert PDF to markdown format."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📖 Opening {pdf_path.name}...")
    
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            
            # Extract metadata
            metadata = reader.metadata
            title = metadata.get('/Title', pdf_path.stem) if metadata else pdf_path.stem
            
            # Clean up title
            if title == pdf_path.stem:
                # Make title more readable
                title = title.replace('-', ' ').replace('_', ' ')
            
            # Start markdown content
            md_content = f"# {title}\n\n"
            
            if metadata:
                if '/Author' in metadata:
                    md_content += f"**Authors:** {metadata['/Author']}\n\n"
                if '/Subject' in metadata:
                    md_content += f"**Subject:** {metadata['/Subject']}\n\n"
                if '/CreationDate' in metadata:
                    date_str = str(metadata['/CreationDate'])
                    md_content += f"**Date:** {date_str}\n\n"
            
            md_content += "---\n\n"
            md_content += "## Abstract\n\n"
            md_content += "_[Abstract will be extracted from first pages]_\n\n"
            md_content += "---\n\n"
            
            # Track sections for better organization
            current_section = ""
            section_pattern = re.compile(r'^(\d+\.?\d*)\s+([A-Z][A-Za-z\s]+)$')
            abstract_found = False
            
            # Extract text from each page
            total_pages = len(reader.pages)
            print(f"📄 Processing {total_pages} pages...")
            
            for i, page in enumerate(reader.pages):
                if i % 10 == 0:
                    print(f"   Page {i+1}/{total_pages}...")
                    
                text = page.extract_text()
                if text.strip():
                    # Clean the text
                    cleaned_text = clean_text(text)
                    
                    # Look for abstract on first few pages
                    if i < 3 and not abstract_found:
                        if 'abstract' in cleaned_text.lower():
                            abstract_found = True
                            # Try to extract abstract
                            abstract_start = cleaned_text.lower().find('abstract')
                            abstract_end = cleaned_text.lower().find('introduction', abstract_start)
                            if abstract_end == -1:
                                abstract_end = cleaned_text.lower().find('1.', abstract_start)
                            if abstract_end == -1:
                                abstract_end = min(abstract_start + 2000, len(cleaned_text))
                            
                            abstract_text = cleaned_text[abstract_start:abstract_end]
                            # Update the abstract section
                            md_content = md_content.replace(
                                "_[Abstract will be extracted from first pages]_",
                                abstract_text
                            )
                    
                    # Try to identify sections
                    lines = cleaned_text.split('\n')
                    page_content = f"\n### Page {i + 1}\n\n"
                    
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                            
                        # Check for section headers
                        section_match = section_pattern.match(line)
                        if section_match:
                            section_num = section_match.group(1)
                            section_title = section_match.group(2)
                            page_content += f"\n## {section_num} {section_title}\n\n"
                            current_section = section_title
                        # Check if line might be a subsection
                        elif line.isupper() and 10 < len(line) < 80:
                            page_content += f"\n### {line.title()}\n\n"
                        # Check for references section
                        elif line.lower().startswith('references') or line.lower() == 'bibliography':
                            page_content += f"\n## References\n\n"
                            current_section = "References"
                        # Regular text
                        else:
                            # Format citations nicely
                            line = re.sub(r'\[(\d+)\]', r'[\1]', line)
                            # Format equations
                            if any(char in line for char in ['∑', '∫', '∂', 'α', 'β', 'γ', 'Σ']):
                                line = f"`{line}`"
                            page_content += line + " "
                    
                    md_content += page_content + "\n\n---\n\n"
            
            # Add footer
            md_content += f"\n## Conversion Notes\n\n"
            md_content += f"- **Source PDF:** `{pdf_path.name}`\n"
            md_content += f"- **Total Pages:** {total_pages}\n"
            md_content += f"- **Converted with:** PyPDF2\n"
            md_content += f"- **Note:** This is an automated conversion. Some formatting, equations, and figures may not be perfectly preserved.\n"
            
            # Save markdown file
            output_file = output_dir / f"{pdf_path.stem}.md"
            output_file.write_text(md_content, encoding='utf-8')
            print(f"✅ Converted {pdf_path.name} → {output_file}")
            
            # Create README for the folder
            readme_path = output_dir / "README.md"
            readme_content = f"""# {pdf_path.stem.replace('-', ' ')} Documentation

This folder contains the converted markdown version of the {pdf_path.stem} paper.

## Files
- `{pdf_path.stem}.md` - Main paper content
- Any extracted figures will be saved here

## Source
- Original PDF: `/literature/pdfs/{pdf_path.name}`
- Converted using PyPDF2 for text extraction

## Paper Information
- **Type:** {"Python package documentation" if "Python" in pdf_path.stem else "Software paper"}
- **Topic:** MNE-Python neuroimaging analysis toolkit

## Note
This is an automated conversion from PDF. Some elements may require manual review:
- Mathematical equations may need reformatting
- Figures are referenced but not embedded
- Tables may need restructuring
- Code blocks may need syntax highlighting

For the most accurate version, please refer to the original PDF.
"""
            readme_path.write_text(readme_content, encoding='utf-8')
            print(f"✅ Created README for {output_dir.name}")
            
    except Exception as e:
        print(f"❌ Error converting {pdf_path.name}: {e}")
        # Create a placeholder file
        error_file = output_dir / f"{pdf_path.stem}_ERROR.md"
        error_content = f"""# Error Converting {pdf_path.name}

## Error Details
{str(e)}

## Suggested Actions
1. Check if the PDF is corrupted
2. Try opening in a PDF reader first
3. Consider using an alternative conversion tool

## Original File
- Path: `{pdf_path}`
"""
        error_file.write_text(error_content, encoding='utf-8')


def main():
    """Convert MNE-related PDFs to markdown."""
    pdfs_dir = Path("/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/literature/pdfs")
    markdown_dir = Path("/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/literature/markdown")
    
    # Define PDFs to convert and their output folders
    conversions = [
        ("MNE-Python.pdf", "MNE-Python"),
        ("MNE-SOFTWARE.pdf", "MNE-SOFTWARE"),
    ]
    
    for pdf_name, folder_name in conversions:
        pdf_path = pdfs_dir / pdf_name
        output_folder = markdown_dir / folder_name
        
        if pdf_path.exists():
            print(f"\n{'='*50}")
            print(f"📄 Processing {pdf_name}...")
            print(f"{'='*50}")
            pdf_to_markdown(pdf_path, output_folder)
        else:
            print(f"❌ {pdf_path} not found")
    
    print("\n" + "="*50)
    print("✅ All conversions complete!")
    print("="*50)
    
    # List what was created
    print("\n📁 Created folders:")
    for folder_name in ["MNE-Python", "MNE-SOFTWARE"]:
        folder_path = markdown_dir / folder_name
        if folder_path.exists():
            files = list(folder_path.glob("*.md"))
            print(f"  - {folder_name}/")
            for f in files:
                print(f"    - {f.name}")


if __name__ == "__main__":
    main()