"""Compare the final PDF/PPTX contents with the measured summary table."""
from pathlib import Path
import csv
import json
import re
import zipfile
import xml.etree.ElementTree as ET
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
with (ROOT / 'results/summary.csv').open() as stream:
    rows = list(csv.DictReader(stream))
pdf = PdfReader(ROOT / 'deliverables/final_report.pdf')
pdf_text = '\n'.join(page.extract_text() or '' for page in pdf.pages)
assert len(pdf.pages) >= 8
assert all((page.extract_text() or '').strip() for page in pdf.pages)
with zipfile.ZipFile(ROOT / 'deliverables/CoupledCF_presentation.pptx') as archive:
    slide_names = sorted(name for name in archive.namelist() if re.fullmatch(r'ppt/slides/slide\d+\.xml', name))
    namespace = {'a': 'http://schemas.openxmlformats.org/drawingml/2006/main'}
    texts = [' '.join(node.text or '' for node in ET.fromstring(archive.read(name)).findall('.//a:t', namespace))
             for name in slide_names]
slide_text = '\n'.join(texts)
for row in rows:
    for metric in ['HR@10', 'NDCG@10']:
        value = f'{float(row[metric + "_mean"]):.4f}'
        assert value in pdf_text, f'Measured mean absent from final PDF: {row["variant"]} {value}'
        assert value in slide_text, f'Measured mean absent from final PPTX table: {row["variant"]} {value}'
assert 'two' in slide_text.lower() and 'one' in slide_text.lower()
assert 'cold' in pdf_text.lower() and 'cold' in slide_text.lower()
assert 'validation' in pdf_text.lower() and 'validation' in slide_text.lower()
receipt = {'status': 'PASS', 'pdf_pages': len(pdf.pages), 'slides': len(slide_names),
           'metric_mean_presence': 'All five HR@10 and NDCG@10 means match measured summary',
           'visual_quality': 'See rendered page/slide inspection; text checks alone do not certify layout'}
(ROOT / '.audit/document_consistency.json').write_text(json.dumps(receipt, indent=2))
print(receipt)
