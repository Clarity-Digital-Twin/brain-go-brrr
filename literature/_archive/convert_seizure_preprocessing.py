#!/usr/bin/env python3
"""Convert seizure_preprocessing.pdf to markdown using marker-pdf."""

from pathlib import Path
import sys

# Try importing marker
try:
    from marker.convert import convert_single_pdf
    from marker.models import load_all_models
except ImportError:
    print("Installing marker dependencies...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "marker-pdf[ocr]"], check=True)
    from marker.convert import convert_single_pdf
    from marker.models import load_all_models

def convert_seizure_preprocessing():
    """Convert seizure_preprocessing.pdf to markdown with images."""
    
    # Setup paths
    pdf_path = Path("/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/literature/pdfs/seizure_preprocessing.pdf")
    output_dir = Path("/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/literature/markdown/seizure_preprocessing")
    
    if not pdf_path.exists():
        print(f"❌ {pdf_path} not found!")
        return
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")
    
    try:
        print("📦 Loading marker models (this may take a minute)...")
        model_lst = load_all_models()
        
        print(f"📖 Converting {pdf_path.name}...")
        print("⏳ This may take a few minutes...")
        
        full_text, images, out_meta = convert_single_pdf(
            str(pdf_path),
            model_lst,
            langs=["English"],
            batch_multiplier=2,
            max_pages=50
        )
        
        # Save markdown
        output_md = output_dir / "seizure_preprocessing.md"
        output_md.write_text(full_text, encoding='utf-8')
        print(f"✅ Saved markdown: {output_md}")
        print(f"   Size: {len(full_text):,} characters")
        
        # Save images if any
        if images:
            print(f"🖼️ Saving {len(images)} images...")
            for img_name, img_data in images.items():
                img_path = output_dir / img_name
                with open(img_path, 'wb') as f:
                    f.write(img_data)
                print(f"   - {img_name}")
        else:
            print("📝 No images found in PDF")
        
        # Save metadata
        import json
        meta_path = output_dir / "metadata.json"
        with open(meta_path, 'w') as f:
            json.dump(out_meta, f, indent=2)
        print(f"📊 Saved metadata: {meta_path}")
        
        print("\n✅ Conversion completed successfully!")
        print(f"📂 Output location: {output_dir}")
        
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you have sufficient memory (marker needs ~4GB)")
        print("2. Try reducing batch_multiplier to 1")
        print("3. Check if the PDF is corrupted")
        raise

if __name__ == "__main__":
    convert_seizure_preprocessing()