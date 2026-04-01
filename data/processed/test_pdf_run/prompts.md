1. PDF to image prompt:
python3 - <<'PY'
from pathlib import Path
from ingest_logic import pdf_slides_to_images

pdf_path = Path("data/uploads/PASTE_YOUR_PDF.pdf")
out_dir = Path("data/processed/test_pdf_run")

paths = pdf_slides_to_images(pdf_path, out_dir, dpi=250)
print("Rendered pages:", len(paths))
print("First image:", paths[0] if paths else None)
PY


2 Image to summary:
python3 - <<'PY'
from pathlib import Path
from vision_engine import visual_summary_from_image

img_path = Path("data/processed/test_pdf_run/slide_044.png")
summary = visual_summary_from_image(img_path, source_id=str(img_path))
print("\n=== Visual Summary ===\n")
print(summary)
PY