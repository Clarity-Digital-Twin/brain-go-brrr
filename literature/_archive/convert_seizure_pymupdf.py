"""Convert SeizureTransformer PDF to Markdown with images using PyMuPDF."""

import pymupdf  # PyMuPDF
from pathlib import Path
import re

def clean_text(text: str) -> str:
    """Clean extracted text."""
    # Fix hyphenations
    text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)
    # Fix ligatures
    text = text.replace('ﬁ', 'fi').replace('ﬂ', 'fl')
    text = text.replace('−', '-').replace('–', '-')
    # Fix spacing
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_with_pymupdf():
    """Extract text and images from SeizureTransformer PDF."""
    
    pdf_path = Path("/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/literature/pdfs/SEIZURE_TRANSFORMER.pdf")
    output_dir = Path("/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/literature/markdown/seizure_transformer")
    
    if not pdf_path.exists():
        print(f"❌ {pdf_path} not found")
        return
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📚 Converting {pdf_path.name} with PyMuPDF...")
    
    try:
        # Open PDF
        doc = pymupdf.open(str(pdf_path))
        
        # Start markdown content
        md_content = "# SeizureTransformer: Scaling U-Net with Transformer for Simultaneous Time-Step Level Seizure Detection from Long EEG Recordings\n\n"
        md_content += "**Authors:** Kerui Wu, Ziyue Zhao, Bülent Yener\n\n"
        md_content += "**Source:** arXiv:2504.00336 (2025)\n\n"
        md_content += "**Conference:** International Conference on Artificial Intelligence in Epilepsy and Other Neurological Disorders (2025)\n\n"
        md_content += "---\n\n"
        
        image_counter = 0
        
        # Process each page
        for page_num, page in enumerate(doc, 1):
            print(f"  📄 Processing page {page_num}/{len(doc)}...")
            
            # Extract text
            text = page.get_text()
            if text.strip():
                # Clean and format text
                cleaned = clean_text(text)
                
                # Detect sections
                if page_num == 1:
                    # Abstract usually on first page
                    if "Abstract" in cleaned or "ABSTRACT" in cleaned:
                        md_content += "## Abstract\n\n"
                        # Extract abstract text
                        abstract_match = re.search(r'Abstract[—-]*(.*?)(?:Index Terms|I\. Introduction|Keywords)', cleaned, re.DOTALL | re.IGNORECASE)
                        if abstract_match:
                            md_content += abstract_match.group(1).strip() + "\n\n"
                
                # Look for section headers
                for match in re.finditer(r'(?:^|\n)([IVX]+\.\s+[A-Z][A-Za-z\s]+)', cleaned):
                    md_content += f"\n## {match.group(1)}\n\n"
                
                # Add the rest of the text
                if page_num > 1:
                    md_content += f"\n### Page {page_num}\n\n"
                    md_content += cleaned + "\n\n"
            
            # Extract images
            image_list = page.get_images(full=True)
            
            for img_index, img in enumerate(image_list):
                try:
                    # Get image data
                    xref = img[0]
                    pix = pymupdf.Pixmap(doc, xref)
                    
                    # Save image
                    if pix.n - pix.alpha < 4:  # GRAY or RGB
                        image_counter += 1
                        img_filename = f"figure_{page_num}_{img_index}.png"
                        img_path = output_dir / img_filename
                        pix.save(str(img_path))
                        print(f"    🖼️ Saved {img_filename}")
                        
                        # Add image reference to markdown
                        md_content += f"\n![Figure {image_counter}]({img_filename})\n\n"
                    
                    pix = None  # Free memory
                    
                except Exception as e:
                    print(f"    ⚠️ Could not extract image {img_index} from page {page_num}: {e}")
        
        # Close document
        doc.close()
        
        # Save markdown
        md_path = output_dir / "SeizureTransformer.md"
        md_path.write_text(md_content, encoding='utf-8')
        
        print(f"\n✅ Conversion complete!")
        print(f"  📝 Markdown: {md_path}")
        print(f"  🖼️ Images: {image_counter} extracted")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    extract_with_pymupdf()