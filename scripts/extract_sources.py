from pathlib import Path
from pypdf import PdfReader

for name, out in [
    ('Group 4_generic frame work for explicit and implicit.pdf', 'paper'),
    ('Research Paper Study and Partial Implementation.pdf', 'assignment'),
]:
    reader = PdfReader(name)
    text = '\n\n'.join(
        f'=== PDF PAGE {i+1} ===\n{page.extract_text(extraction_mode="layout")}'
        for i, page in enumerate(reader.pages)
    )
    Path(f'sources/{out}.txt').write_text(text, encoding='utf-8')
    print(out, len(reader.pages), len(text), flush=True)
