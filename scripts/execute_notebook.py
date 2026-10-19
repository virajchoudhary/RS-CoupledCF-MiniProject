"""Execute the notebook with a kernel explicitly pinned to this Python."""
from pathlib import Path
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
(ROOT / '.audit/tmp').mkdir(parents=True, exist_ok=True)
os.environ['TEMP'] = str(ROOT / '.audit/tmp')
os.environ['TMP'] = str(ROOT / '.audit/tmp')
import nbformat
from nbclient import NotebookClient

kernel_root = ROOT / '.audit/jupyter/kernels'
kernel = kernel_root / 'coupledcf'
kernel.mkdir(parents=True, exist_ok=True)
(kernel / 'kernel.json').write_text(json.dumps({
    'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
    'display_name': 'CoupledCF project', 'language': 'python',
}), encoding='utf-8')
os.environ['JUPYTER_PATH'] = str(kernel_root.parent)
os.environ['JUPYTER_RUNTIME_DIR'] = str(ROOT / '.audit/jupyter/runtime')
os.environ['IPYTHONDIR'] = str(ROOT / '.audit/ipython')
Path(os.environ['JUPYTER_RUNTIME_DIR']).mkdir(parents=True, exist_ok=True)
path = ROOT / 'notebooks/CoupledCF_reproduction.ipynb'
nb = nbformat.read(path, as_version=4)
client = NotebookClient(nb, timeout=600, kernel_name='coupledcf', resources={'metadata': {'path': str(ROOT)}})
client.execute()
destination = ROOT / 'notebooks/CoupledCF_reproduction.executed.ipynb'
nbformat.write(nb, destination)
code_cells = [cell for cell in nb.cells if cell.cell_type == 'code']
assert all(cell.execution_count is not None for cell in code_cells)
assert not any(output.output_type == 'error' for cell in code_cells for output in cell.outputs)
receipt = {'status': 'PASS', 'executed_code_cells': len(code_cells), 'kernel_python': sys.executable,
           'notebook': str(destination.relative_to(ROOT))}
(ROOT / '.audit/notebook_execution.json').write_text(json.dumps(receipt, indent=2))
print(receipt)
