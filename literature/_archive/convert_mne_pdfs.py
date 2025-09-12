"""Convert MNE PDFs to Markdown using marker tool."""

import subprocess
from pathlib import Path
import shutil


def convert_with_marker(pdf_path: Path, output_dir: Path) -> None:
    """Convert PDF to markdown using marker tool."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run marker command
    cmd = [
        "marker_single",
        str(pdf_path),
        str(output_dir),
        "--batch_multiplier", "2",
        "--max_pages", "100"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Converted {pdf_path.name} → {output_dir}")
            
            # Rename the output file to match the PDF name
            # Marker creates files with specific naming, we want consistent naming
            for md_file in output_dir.glob("*.md"):
                if md_file.name != f"{pdf_path.stem}.md":
                    new_name = output_dir / f"{pdf_path.stem}.md"
                    shutil.move(str(md_file), str(new_name))
                    print(f"   Renamed to {new_name.name}")
        else:
            print(f"❌ Failed to convert {pdf_path.name}")
            print(f"   Error: {result.stderr}")
    except Exception as e:
        print(f"❌ Error converting {pdf_path.name}: {e}")


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
            print(f"\n📄 Processing {pdf_name}...")
            convert_with_marker(pdf_path, output_folder)
        else:
            print(f"❌ {pdf_path} not found")
    
    print("\n✅ Conversion complete!")
    
    # Create README for MNE folders if they don't exist
    for folder_name in ["MNE-Python", "MNE-SOFTWARE"]:
        folder_path = markdown_dir / folder_name
        readme_path = folder_path / "README.md"
        
        if folder_path.exists() and not readme_path.exists():
            readme_content = f"""# {folder_name} Documentation

This folder contains the converted markdown version of the {folder_name} PDF paper.

## Source
- Original PDF: `/literature/pdfs/{folder_name}.pdf`
- Converted using marker tool for better text extraction

## Contents
- Main paper: `{folder_name}.md`
- Figures: Any extracted figures are in this directory

## Note
This is an automated conversion. Some formatting may not be perfect, but the content is preserved for reference.
"""
            readme_path.write_text(readme_content)
            print(f"✅ Created README for {folder_name}")


if __name__ == "__main__":
    main()