import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import { fileURLToPath, pathToFileURL } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const packagePath = path.join(root, '.audit/presentation/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs');
const { PresentationFile, FileBlob } = await import(pathToFileURL(packagePath).href);
const source = path.join(root, 'deliverables/CoupledCF_presentation.pptx');
const output = path.join(root, '.audit/presentation/final/roundtrip_rendered');
await fs.mkdir(output, { recursive: true });
const presentation = await PresentationFile.importPptx(await FileBlob.load(source));
const files = [];
for (const [index, slide] of presentation.slides.items.entries()) {
  const filename = `slide-${String(index + 1).padStart(2, '0')}.png`;
  const blob = await presentation.export({ slide, format: 'png', scale: 1 });
  await fs.writeFile(path.join(output, filename), new Uint8Array(await blob.arrayBuffer()));
  files.push(filename);
  console.log(`Rendered finalized PPTX slide ${index + 1}`, filename);
}
const digest = crypto.createHash('sha256').update(await fs.readFile(source)).digest('hex');
await fs.writeFile(path.join(output, 'render_receipt.json'), JSON.stringify({
  status: 'PASS', pptx_sha256: digest, slides: files.length, files,
  method: 'Imported the finalized PPTX back into Artifact Tool and rendered sequentially',
  limitation: 'This is a render check, not an inspection inside the PowerPoint application',
}, null, 2));
