"""Convert SeizureTransformer PDF to Markdown with images using marker."""

from pathlib import Path
import subprocess
import shutil
import os

def convert_with_marker():
    """Use marker to convert SeizureTransformer PDF with images."""
    
    pdfs_dir = Path("/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/literature/pdfs")
    output_base = Path("/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/literature/markdown")
    
    seizure_pdf = pdfs_dir / "SEIZURE_TRANSFORMER.pdf"
    
    if not seizure_pdf.exists():
        print(f"❌ {seizure_pdf} not found")
        return
    
    # Create output directory
    output_dir = output_base / "seizure_transformer"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📚 Converting {seizure_pdf.name} with marker...")
    print("⏳ This may take a few minutes...")
    
    try:
        # Run marker command
        # marker converts to a folder with the same name as the PDF
        cmd = [
            "marker_single",
            str(seizure_pdf),
            str(output_dir),
            "--batch_multiplier", "2",
            "--langs", "English"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ Marker failed: {result.stderr}")
            # Try alternative command format
            cmd = [
                "marker",
                str(seizure_pdf),
                "-o", str(output_dir),
                "--langs", "English"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"❌ Alternative marker command also failed: {result.stderr}")
                return
        
        print(f"✅ Conversion completed!")
        
        # Check what was created
        for item in output_dir.iterdir():
            print(f"  📄 Created: {item.name}")
            
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        print("\n🔧 Trying manual marker import...")
        
        try:
            from marker.convert import convert_single_pdf
            from marker.models import load_all_models
            
            print("📦 Loading marker models...")
            model_lst = load_all_models()
            
            print(f"📖 Converting {seizure_pdf.name}...")
            full_text, images, out_meta = convert_single_pdf(
                str(seizure_pdf),
                model_lst,
                langs=["English"],
                batch_multiplier=2
            )
            
            # Save markdown
            output_md = output_dir / "SeizureTransformer.md"
            output_md.write_text(full_text, encoding='utf-8')
            print(f"✅ Saved markdown: {output_md}")
            
            # Save images if any
            if images:
                for img_name, img_data in images.items():
                    img_path = output_dir / img_name
                    with open(img_path, 'wb') as f:
                        f.write(img_data)
                    print(f"🖼️ Saved image: {img_name}")
            
            print(f"✅ Conversion completed successfully!")
            
        except Exception as e2:
            print(f"❌ Manual import also failed: {e2}")
            print("\n💡 Please try running manually:")
            print(f"   cd {pdfs_dir}")
            print(f"   marker SEIZURE_TRANSFORMER.pdf")

if __name__ == "__main__":
    convert_with_marker()