import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const AUDIT_DIR = path.join(ROOT, ".audit", "presentation");
const SKILL_DIR = "C:\\Users\\viraj\\.codex\\plugins\\cache\\openai-primary-runtime\\presentations\\26.1015.12040\\skills\\presentations";
const RUNTIME_NODE_MODULES = process.env.CODEX_RUNTIME_NODE_MODULES
  ?? "C:\\Users\\viraj\\.cache\\codex-runtimes\\codex-primary-runtime\\dependencies\\node\\node_modules";
process.env.RUNTIME_NODE_MODULES ??= RUNTIME_NODE_MODULES;
const RUNTIME_PYTHON = process.env.CODEX_RUNTIME_PYTHON
  ?? "C:\\Users\\viraj\\.cache\\codex-runtimes\\codex-primary-runtime\\dependencies\\python\\python.exe";
const ARTIFACT_TOOL_PATH = path.join(
  AUDIT_DIR,
  "node_modules",
  "@oai",
  "artifact-tool",
  "dist",
  "artifact_tool.mjs",
);
const PAPER_REFERENCE = {
  deepcf: { hr: 0.7147, ndcg: 0.4312 },
  local: { hr: 0.8212, ndcg: 0.5408 },
  global: { hr: 0.7826, ndcg: 0.5252 },
  coupled: { hr: 0.8252, ndcg: 0.5440 },
};
const EXPECTED_TABLE7_KEYS = Object.keys(PAPER_REFERENCE);
const W = 1280;
const H = 720;
const EXPECTED_SLIDE_SIZE_EMU = "12192000,6858000";
const OUTPUT_SLIDE_COUNT = 16;
// Excel stores 15 significant digits; chart snapshots need only display precision.
const chartNumber = (value) => Number(value.toPrecision(12));
const PALETTE = Object.freeze({
  paper: "#FFFFFF",
  ink: "#15232D",
  body: "#293A45",
  muted: "#5F6D76",
  line: "#D5DEE3",
  blue: "#19689A",
  bluePale: "#EAF2F7",
  teal: "#147C74",
  tealPale: "#E7F3F0",
  orange: "#C97725",
  orangePale: "#FBF0E2",
  red: "#9B4C51",
  redPale: "#F7EBEC",
  greyPale: "#F3F6F7",
});

const args = new Set(process.argv.slice(2));
if (args.has("--help")) {
  console.log("Usage: node scripts/build_presentation.mjs --draft | --final");
  process.exit(0);
}
if (args.size !== 1 || (!args.has("--draft") && !args.has("--final"))) {
  throw new Error("Choose exactly one mode: --draft or --final.");
}
const MODE = args.has("--final") ? "final" : "draft";

function invariant(condition, message) {
  if (!condition) throw new Error(message);
}

async function exists(filePath) {
  return await fs.access(filePath).then(() => true, () => false);
}

async function ensureDirectory(directory) {
  await fs.mkdir(directory, { recursive: true });
}

async function ensureModuleJunction() {
  await ensureDirectory(AUDIT_DIR);
  const linkPath = path.join(AUDIT_DIR, "node_modules");
  if (await exists(linkPath)) {
    const [actual, expected] = await Promise.all([
      fs.realpath(linkPath),
      fs.realpath(RUNTIME_NODE_MODULES),
    ]);
    invariant(
      actual.toLowerCase() === expected.toLowerCase(),
      `The existing module link points to ${actual}, expected ${expected}.`,
    );
    return;
  }
  await fs.symlink(RUNTIME_NODE_MODULES, linkPath, process.platform === "win32" ? "junction" : "dir");
}

function parseCsv(text, label) {
  const rows = [];
  let row = [];
  let field = "";
  let quoted = false;
  for (let index = 0; index < text.length; index += 1) {
    const char = text[index];
    if (quoted) {
      if (char === '"') {
        if (text[index + 1] === '"') {
          field += '"';
          index += 1;
        } else {
          quoted = false;
        }
      } else {
        field += char;
      }
    } else if (char === '"' && field.length === 0) {
      quoted = true;
    } else if (char === ",") {
      row.push(field);
      field = "";
    } else if (char === "\n" || char === "\r") {
      row.push(field);
      field = "";
      if (row.some((value) => value.length > 0)) rows.push(row);
      row = [];
      if (char === "\r" && text[index + 1] === "\n") index += 1;
    } else {
      field += char;
    }
  }
  invariant(!quoted, `${label} ends inside a quoted CSV field.`);
  row.push(field);
  if (row.some((value) => value.length > 0)) rows.push(row);
  invariant(rows.length > 0, `${label} is empty.`);
  const headers = rows[0].map((value) => value.trim());
  invariant(headers.every(Boolean), `${label} has a blank column name.`);
  invariant(new Set(headers).size === headers.length, `${label} has duplicate column names.`);
  return rows.slice(1).map((values, index) => {
    invariant(
      values.length === headers.length,
      `${label} row ${index + 2} has ${values.length} cells; expected ${headers.length}.`,
    );
    return Object.fromEntries(headers.map((header, column) => [header, values[column].trim()]));
  });
}

function numberFrom(value, label) {
  const number = Number(value);
  invariant(Number.isFinite(number), `${label} must be a finite number; received ${value}.`);
  return number;
}

function formatDecimal(value, digits = 4) {
  return numberFrom(value, "Displayed metric").toFixed(digits);
}

function formatMetric(mean, deviation, digits = 4) {
  return `${formatDecimal(mean, digits)} +/- ${formatDecimal(deviation, digits)}`;
}

function mean(values) {
  invariant(values.length > 0, "Cannot average an empty list.");
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function sampleStd(values) {
  invariant(values.length >= 2, "Sample SD requires at least two values.");
  const center = mean(values);
  return Math.sqrt(values.reduce((sum, value) => sum + ((value - center) ** 2), 0) / (values.length - 1));
}

function csvSummaryValue(row, key) {
  return numberFrom(row[key], `summary.csv ${key}`);
}

function variantLabel(variant) {
  return ({
    deepcf: "DeepCF",
    local: "Local CNN",
    global: "Global",
    coupled_local: "Local + implicit",
    coupled: "CoupledCF",
  })[variant] ?? variant;
}

function metricStatsForVariant(bundle, variant, key) {
  const values = bundle.runs
    .filter((row) => row.variant === variant)
    .map((row) => numberFrom(row[key], `runs.csv ${variant} ${key}`));
  return { mean: mean(values), std: sampleStd(values) };
}

function ablationStats(bundle, comparison, column) {
  const values = bundle.ablations
    .filter((row) => row.comparison === comparison)
    .map((row) => numberFrom(row[column], `ablation.csv ${comparison} ${column}`));
  return { mean: mean(values), std: sampleStd(values) };
}

function validateTable7(references) {
  const paper = references.paper_table7_pdf_page11;
  invariant(paper && typeof paper === "object", "references.json is missing paper_table7_pdf_page11.");
  for (const variant of EXPECTED_TABLE7_KEYS) {
    const actual = paper[variant];
    invariant(actual, `Table 7 reference is missing ${variant}.`);
    const expected = PAPER_REFERENCE[variant];
    invariant(
      actual["HR@10"] === expected.hr && actual["NDCG@10"] === expected.ndcg,
      `Table 7 reference for ${variant} differs from the source-checked value.`,
    );
  }
}

function validateBundle(bundle, config) {
  const configuredVariants = config.variants;
  const configuredSeeds = config.seeds;
  invariant(Array.isArray(configuredVariants) && configuredVariants.length > 0, "main.json needs variants.");
  invariant(Array.isArray(configuredSeeds) && configuredSeeds.length >= 2, "main.json needs at least two seeds.");
  invariant(bundle.summary.length === configuredVariants.length, "summary.csv variant count differs from main.json.");
  invariant(bundle.runs.length === configuredVariants.length * configuredSeeds.length, "runs.csv does not contain every configured model-seed run.");

  const configuredPairs = new Set();
  for (const variant of configuredVariants) {
    for (const seed of configuredSeeds) configuredPairs.add(`${variant}:${seed}`);
  }
  for (const run of bundle.runs) {
    invariant(configuredPairs.delete(`${run.variant}:${run.seed}`), `Unexpected or duplicate run ${run.variant}/${run.seed}.`);
    for (const metric of ["HR@10", "NDCG@10"]) {
      const value = numberFrom(run[metric], `runs.csv ${metric}`);
      invariant(value >= 0 && value <= 1, `runs.csv ${metric} must be between 0 and 1.`);
    }
    invariant(numberFrom(run.parameters, "runs.csv parameters") > 0, "Every run needs a positive parameter count.");
  }
  invariant(configuredPairs.size === 0, `Missing runs: ${[...configuredPairs].join(", ")}.`);

  const summaryByVariant = new Map(bundle.summary.map((row) => [row.variant, row]));
  invariant(summaryByVariant.size === configuredVariants.length, "summary.csv has duplicate model rows.");
  for (const variant of configuredVariants) {
    const row = summaryByVariant.get(variant);
    invariant(row, `summary.csv is missing ${variant}.`);
    invariant(numberFrom(row.runs, "summary.csv runs") === configuredSeeds.length, `${variant} has the wrong run count.`);
    for (const [metricMeanKey, metricStdKey, runKey] of [
      ["HR@10_mean", "HR@10_std", "HR@10"],
      ["NDCG@10_mean", "NDCG@10_std", "NDCG@10"],
    ]) {
      const calculated = metricStatsForVariant(bundle, variant, runKey);
      invariant(
        Math.abs(csvSummaryValue(row, metricMeanKey) - calculated.mean) < 1e-9,
        `summary.csv ${metricMeanKey} for ${variant} does not match runs.csv.`,
      );
      invariant(
        Math.abs(csvSummaryValue(row, metricStdKey) - calculated.std) < 1e-9,
        `summary.csv ${metricStdKey} for ${variant} does not match runs.csv.`,
      );
    }
  }

  const expectedComparisons = [
    { comparison: "add explicit local and global", first: "coupled", second: "deepcf" },
    { comparison: "add global to local plus implicit", first: "coupled", second: "coupled_local" },
    { comparison: "add local to implicit", first: "coupled_local", second: "deepcf" },
    { comparison: "local-only versus global-only", first: "local", second: "global" },
  ];
  const runByPair = new Map(bundle.runs.map((row) => [`${row.variant}:${row.seed}`, row]));
  invariant(bundle.ablations.length === expectedComparisons.length * configuredSeeds.length, "Ablation rows do not match comparison coverage.");
  for (const comparison of expectedComparisons) {
    for (const seed of configuredSeeds) {
      const row = bundle.ablations.find((item) =>
        item.comparison === comparison.comparison && Number(item.seed) === seed,
      );
      invariant(row, `ablation.csv is missing ${comparison.comparison}, seed ${seed}.`);
      invariant(row.first === comparison.first && row.second === comparison.second, `Unexpected variants for ${comparison.comparison}.`);
      const first = runByPair.get(`${comparison.first}:${seed}`);
      const second = runByPair.get(`${comparison.second}:${seed}`);
      for (const [deltaKey, metricKey] of [["delta_HR@10", "HR@10"], ["delta_NDCG@10", "NDCG@10"]]) {
        const difference = numberFrom(first[metricKey], `runs.csv ${metricKey}`)
          - numberFrom(second[metricKey], `runs.csv ${metricKey}`);
        invariant(
          Math.abs(numberFrom(row[deltaKey], `ablation.csv ${deltaKey}`) - difference) < 1e-9,
          `${comparison.comparison} ${deltaKey} does not match the paired run records.`,
        );
      }
    }
  }

  validateTable7(bundle.references);
  invariant(bundle.references.split_sha256, "references.json is missing the data split hash.");
  invariant(bundle.manifest.split_sha256 === bundle.references.split_sha256, "Result and dataset split hashes differ.");
  invariant(bundle.manifest.dataset === "MovieLens 1M", "Dataset manifest does not identify MovieLens 1M.");
  invariant(bundle.manifest.leakage_checks === "PASS", "Dataset leakage checks are not marked PASS.");
  return bundle;
}

async function readBundle() {
  const paths = {
    summary: path.join(ROOT, "results", "summary.csv"),
    runs: path.join(ROOT, "results", "runs.csv"),
    ablations: path.join(ROOT, "results", "ablation.csv"),
    references: path.join(ROOT, "results", "references.json"),
    config: path.join(ROOT, "configs", "main.json"),
    student: path.join(ROOT, "configs", "student.json"),
    manifest: path.join(ROOT, "data", "processed", "manifest.json"),
  };
  const required = ["summary", "runs", "ablations", "references", "config", "student", "manifest"];
  const present = Object.fromEntries(await Promise.all(required.map(async (key) =>
    [key, await exists(paths[key])])));
  if (MODE === "final") {
    const missing = required.filter((key) => !present[key]);
    invariant(missing.length === 0, `Final mode requires complete inputs; missing: ${missing.join(", ")}.`);
  }

  const loaded = { paths, present, config: null, student: null, manifest: null, summary: null, runs: null, ablations: null, references: null };
  for (const key of required) {
    if (!present[key]) continue;
    const text = await fs.readFile(paths[key], "utf8");
    if (key === "config" || key === "student" || key === "manifest" || key === "references") {
      loaded[key] = JSON.parse(text);
    } else {
      loaded[key] = parseCsv(text, path.basename(paths[key]));
    }
  }

  const resultsReady = required.every((key) => present[key])
    && loaded.summary !== null
    && loaded.runs !== null
    && loaded.ablations !== null
    && loaded.references !== null;
  if (resultsReady) validateBundle(loaded, loaded.config);
  if (MODE === "final") invariant(resultsReady, "Final mode requires a complete, validated result bundle.");
  return { ...loaded, resultsReady };
}

async function loadArtifactTool() {
  await ensureModuleJunction();
  const [{ Presentation, PresentationFile }, utility] = await Promise.all([
    import(pathToFileURL(ARTIFACT_TOOL_PATH).href),
    import(pathToFileURL(path.join(SKILL_DIR, "container_tools", "artifact_tool_utils.mjs")).href),
  ]);
  const fontFamily = utility.resolvePresentationFont({ fontFamily: "Arial" });
  return {
    Presentation,
    PresentationFile,
    fontFamily,
    finalizePresentation: utility.finalizePresentation,
    applyPresentationChartFont: utility.applyPresentationChartFont,
    makeNativeBulletParagraphs: utility.makeNativeBulletParagraphs,
  };
}

function addText(slide, text, position, style = {}, name = "text") {
  const shape = slide.shapes.add({
    geometry: "textbox",
    name,
    position,
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  shape.text = String(text);
  shape.text.style = {
    typeface: FONT,
    fontSize: 23,
    color: PALETTE.body,
    alignment: "left",
    verticalAlignment: "top",
    wrap: "square",
    autoFit: "shrinkText",
    insets: 0,
    ...style,
  };
  return shape;
}

let FONT = "Arial";

function addBullets(slide, items, position, options = {}) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    name: options.name ?? "evidence-list",
    position,
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  shape.text = options.makeNativeBulletParagraphs(items, {
    marginLeftPoints: options.marginLeftPoints ?? 19,
    hangingPoints: options.hangingPoints ?? 10,
    spaceAfterPoints: options.spaceAfterPoints ?? 11,
  });
  shape.text.style = {
    typeface: FONT,
    fontSize: options.fontSize ?? 23,
    color: options.color ?? PALETTE.body,
    verticalAlignment: "top",
    wrap: "square",
    autoFit: "shrinkText",
    insets: 0,
  };
  return shape;
}

function addRule(slide, left, top, width, fill = PALETTE.line, height = 1) {
  return slide.shapes.add({
    geometry: "rect",
    name: "divider",
    position: { left, top, width, height },
    fill,
    line: { style: "solid", fill, width: 0 },
  });
}

function addTitle(presentation, slide, title, subtitle) {
  slide.background.fill = PALETTE.paper;
  addText(slide, title, { left: 64, top: 42, width: 1152, height: 56 }, {
    typeface: FONT,
    fontSize: 44,
    bold: true,
    color: PALETTE.ink,
    autoFit: "shrinkText",
  }, "slide-title");
  addRule(slide, 64, 108, 1152, PALETTE.blue, 3);
  if (subtitle) {
    addText(slide, subtitle, { left: 64, top: 122, width: 1152, height: 48 }, {
      fontSize: 23,
      color: PALETTE.muted,
    }, "slide-subtitle");
  }
  return slide;
}

function addNode(slide, label, position, options = {}) {
  const shape = slide.shapes.add({
    geometry: "rect",
    name: options.name ?? "architecture-node",
    position,
    fill: options.fill ?? PALETTE.bluePale,
    line: { style: "solid", fill: options.stroke ?? PALETTE.line, width: 1.5 },
  });
  shape.text = label;
  shape.text.style = {
    typeface: FONT,
    fontSize: options.fontSize ?? 21,
    bold: options.bold ?? false,
    color: options.color ?? PALETTE.ink,
    alignment: "center",
    verticalAlignment: "middle",
    wrap: "square",
    autoFit: "shrinkText",
    insets: { top: 8, right: 10, bottom: 8, left: 10 },
  };
  return shape;
}

function connect(slide, from, to, options = {}) {
  return slide.shapes.connect(from, to, {
    kind: options.kind ?? "elbow",
    fromSide: options.fromSide,
    toSide: options.toSide,
    line: {
      style: options.style ?? "solid",
      fill: options.color ?? PALETTE.muted,
      width: options.width ?? 2,
    },
    head: { type: "none" },
    tail: options.arrow === false ? { type: "none" } : { type: "triangle", width: "sm", length: "sm" },
  });
}

function addNativeTable(slide, values, position, options = {}) {
  const rowCount = values.length;
  const columnCount = values[0]?.length ?? 0;
  invariant(rowCount > 0 && columnCount > 0, "A native table requires values.");
  invariant(values.every((row) => row.length === columnCount), "Native table rows must have the same number of cells.");
  const table = slide.tables.add({
    rows: rowCount,
    columns: columnCount,
    left: position.left,
    top: position.top,
    width: position.width,
    height: position.height,
    columnWidths: options.columnWidths,
    values,
  });
  table.styleOptions = { headerRow: false, firstColumn: false, bandedRows: false, bandedColumns: false };
  table.borders.assign({ style: "solid", fill: PALETTE.line, width: 1 });
  const headerFill = options.headerFill ?? PALETTE.ink;
  const bodyFontSize = options.fontSize ?? 17;
  for (let row = 0; row < rowCount; row += 1) {
    table.rows[row].height = options.rowHeights?.[row] ?? position.height / rowCount;
    for (let column = 0; column < columnCount; column += 1) {
      const cell = table.getCell(row, column);
      const isHeader = row === 0;
      const isKeyColumn = options.boldFirstColumn && column === 0;
      cell.fill = isHeader ? headerFill : (row % 2 === 0 ? PALETTE.greyPale : PALETTE.paper);
      cell.text.style = {
        typeface: FONT,
        fontSize: isHeader ? bodyFontSize : bodyFontSize,
        bold: isHeader || isKeyColumn,
        color: isHeader ? PALETTE.paper : PALETTE.body,
        alignment: column === 0 ? "left" : "center",
        verticalAlignment: "middle",
        autoFit: "shrinkText",
        wrap: "square",
        insets: { top: 5, right: 6, bottom: 5, left: 8 },
      };
    }
  }
  return table;
}

function addPaperCitation(slide, pages, detail = "") {
  const pageText = Array.isArray(pages) ? pages.join(", ") : pages;
  slide.speakerNotes.text = [
    "Zhang, Q., Wang, L., Meng, X., Xu, K., and Hu, J. (2019). A Generic Framework for Learning Explicit and Implicit User-Item Couplings in Recommendation. IEEE Access, 7, 123944-123958. https://doi.org/10.1109/ACCESS.2019.2937841",
    `Relevant printed IEEE page(s): ${pageText}. The supplied 15-page paper PDF is an image-only copy; page locators here use its printed IEEE pagination.`,
    detail,
  ].filter(Boolean).join("\n");
}

function addProjectNotes(slide, files, extra = "") {
  slide.speakerNotes.text = [
    `Project evidence: ${files.join(", ")}.`,
    extra,
  ].filter(Boolean).join("\n");
}

function heading(slide, text, left, top, width, color = PALETTE.blue) {
  return addText(slide, text, { left, top, width, height: 34 }, {
    fontSize: 23,
    bold: true,
    color,
  }, "section-heading");
}

function addPlainColumn(slide, title, body, position, accent) {
  addRule(slide, position.left, position.top, 44, accent, 4);
  addText(slide, title, { left: position.left, top: position.top + 18, width: position.width, height: 42 }, {
    fontSize: 26,
    bold: true,
    color: PALETTE.ink,
  }, "column-title");
  addText(slide, body, { left: position.left, top: position.top + 72, width: position.width, height: position.height - 72 }, {
    fontSize: position.fontSize ?? 22,
    lineSpacing: 1.04,
  }, "column-copy");
}

function makeSlide1(presentation, bundle) {
  const slide = presentation.slides.add();
  slide.background.fill = PALETTE.paper;
  addRule(slide, 64, 80, 68, PALETTE.teal, 6);
  addText(slide, "RECOMMENDATION SYSTEMS MINI PROJECT", { left: 64, top: 112, width: 900, height: 34 }, {
    fontSize: 20,
    bold: true,
    color: PALETTE.blue,
  }, "cover-kicker");
  addText(slide, "CoupledCF on MovieLens 1M", { left: 64, top: 190, width: 1120, height: 104 }, {
    fontSize: 62,
    bold: true,
    color: PALETTE.ink,
    autoFit: "shrinkText",
  }, "cover-title");
  addText(slide, "Attribute-based explicit coupling for Top-N recommendation", { left: 64, top: 314, width: 1100, height: 58 }, {
    fontSize: 31,
    color: PALETTE.teal,
  }, "cover-subtitle");
  addRule(slide, 64, 412, 1152, PALETTE.line, 2);
  addText(slide, "Partial reproduction of Zhang et al. (2019)", { left: 64, top: 444, width: 920, height: 42 }, {
    fontSize: 25,
    color: PALETTE.body,
  }, "cover-paper");
  const institution = bundle.student?.institution ?? "[Institution]";
  const group = bundle.student?.group ?? "[Group number]";
  const course = bundle.student?.course ?? "Recommendation Systems";
  addText(slide, `${course}  |  ${group}  |  ${institution}`, { left: 64, top: 514, width: 1120, height: 42 }, {
    fontSize: 21,
    color: PALETTE.muted,
  }, "cover-details");
  slide.speakerNotes.text = "The paper studies explicit and implicit user-item couplings. This project evaluates the attribute-based Top-N component as a partial reproduction on MovieLens 1M.\n\nZhang, Q., Wang, L., Meng, X., Xu, K., and Hu, J. (2019). IEEE Access, 7, 123944-123958. https://doi.org/10.1109/ACCESS.2019.2937841";
}

function makeSlide2(presentation) {
  const slide = addTitle(presentation, presentation.slides.add(), "The recommendation problem", "A personalized Top-N ranking task with sparse user histories");
  addPlainColumn(
    slide,
    "User need",
    "A catalogue can contain thousands of items, while each user interacts with only a small fraction. A useful system must rank unseen items from limited feedback.",
    { left: 72, top: 204, width: 470, height: 336 },
    PALETTE.blue,
  );
  addPlainColumn(
    slide,
    "Paper framing",
    "The paper treats recommendation as learning user-item interactions. It evaluates both Top-N ranking and rating prediction, with sparsity and cold-start as central concerns.",
    { left: 680, top: 204, width: 490, height: 336 },
    PALETTE.teal,
  );
  addRule(slide, 72, 574, 1096, PALETTE.line, 1);
  addText(slide, "This project studies the attribute-based Top-N component and reports HR@10 and NDCG@10.", {
    left: 72, top: 598, width: 1100, height: 52,
  }, { fontSize: 23, bold: true, color: PALETTE.ink }, "scope-statement");
  addPaperCitation(slide, "123944-123945", "The paper's abstract and introduction frame recommendation around non-IID user-item relations, sparse interaction histories, and personalized item ranking.");
}

function makeSlide3(presentation) {
  const slide = addTitle(presentation, presentation.slides.add(), "Research gap and contribution", "The paper combines side-information coupling with ID-based interaction learning");
  addPlainColumn(
    slide,
    "Research gap",
    "ID-only interaction models can underuse attributes and text that describe users and items. Side information also creates many feature interactions that simple concatenation may leave unmodeled.",
    { left: 72, top: 208, width: 490, height: 326 },
    PALETTE.orange,
  );
  addPlainColumn(
    slide,
    "Authors' contribution",
    "CoupledCF learns local explicit interactions with CNNs, adds a global product representation, and integrates these features with the implicit DeepCF branch.",
    { left: 680, top: 208, width: 490, height: 326 },
    PALETTE.teal,
  );
  addRule(slide, 72, 572, 1096, PALETTE.line, 1);
  addText(slide, "Project contribution: implement and compare five branch configurations under one matched MovieLens 1M ranking protocol.", {
    left: 72, top: 598, width: 1096, height: 58,
  }, { fontSize: 23, bold: true, color: PALETTE.ink }, "project-contribution");
  addPaperCitation(slide, "123944-123946", "The authors describe CoupledCF's motivation and claimed framework contributions in the abstract and introduction. Project results on this deck are kept separate from the paper's claims.");
}

function makeSlide4(presentation, bulletHelper) {
  const slide = addTitle(presentation, presentation.slides.add(), "CoupledCF in the paper", "Paper-level view: explicit feature coupling and implicit ID interaction meet at the output layer");
  const sideInfo = addNode(slide, "User and item\nattributes\nReview-derived features", { left: 70, top: 214, width: 224, height: 126 }, {
    fill: PALETTE.bluePale, stroke: PALETTE.blue, fontSize: 21, name: "paper-explicit-inputs",
  });
  const ids = addNode(slide, "User ID and\nitem ID", { left: 70, top: 430, width: 224, height: 108 }, {
    fill: PALETTE.orangePale, stroke: PALETTE.orange, fontSize: 23, name: "paper-implicit-inputs",
  });
  const explicit = addNode(slide, "Explicit coupling\nLocal CNN features\nGlobal product features", { left: 374, top: 204, width: 262, height: 150 }, {
    fill: PALETTE.tealPale, stroke: PALETTE.teal, fontSize: 21, name: "paper-explicit-path",
  });
  const implicit = addNode(slide, "DeepCF\nID embeddings\nElement product + MLP", { left: 374, top: 419, width: 262, height: 128 }, {
    fill: PALETTE.orangePale, stroke: PALETTE.orange, fontSize: 21, name: "paper-implicit-path",
  });
  const fusion = addNode(slide, "Concatenate\nand learn a\njoint representation", { left: 714, top: 304, width: 206, height: 160 }, {
    fill: PALETTE.greyPale, stroke: PALETTE.blue, fontSize: 21, name: "paper-fusion",
  });
  const output = addNode(slide, "Recommendation\nscore\nTop-N or rating", { left: 988, top: 318, width: 224, height: 130 }, {
    fill: PALETTE.bluePale, stroke: PALETTE.blue, fontSize: 21, name: "paper-output",
  });
  connect(slide, sideInfo, explicit, { fromSide: "right", toSide: "left" });
  connect(slide, ids, implicit, { fromSide: "right", toSide: "left" });
  connect(slide, explicit, fusion, { fromSide: "right", toSide: "left" });
  connect(slide, implicit, fusion, { fromSide: "right", toSide: "left" });
  connect(slide, fusion, output, { fromSide: "right", toSide: "left", kind: "straight" });
  addText(slide, "Paper framework", { left: 72, top: 178, width: 240, height: 28 }, {
    fontSize: 18, bold: true, color: PALETTE.muted,
  }, "diagram-label");
  slide.speakerNotes.text = "This is a simplified editable rendering of the paper's framework. The project below uses MovieLens attributes for the explicit branch; it does not implement the review-text pathway.\n\nZhang, Q., Wang, L., Meng, X., Xu, K., and Hu, J. (2019). IEEE Access, 7, 123944-123958. https://doi.org/10.1109/ACCESS.2019.2937841\nRelevant printed IEEE pages: 123945-123950. See the claimed contributions, Figure 1, and Figures 2-3.";
}

function makeSlide5(presentation, bundle, bulletHelper) {
  const slide = addTitle(presentation, presentation.slides.add(), "Feature inputs and representations", "The project uses profile and genre metadata for explicit coupling");
  const userBlock = addNode(slide, "User profile fields\nGender\nAge group\nOccupation\nZIP code", { left: 72, top: 212, width: 245, height: 254 }, {
    fill: PALETTE.bluePale, stroke: PALETTE.blue, fontSize: 21, name: "user-profile-fields",
  });
  const userEmbed = addNode(slide, "Four learned\n8-dimensional\nfield embeddings\nConcatenate, then\nproject to 8D", { left: 374, top: 225, width: 250, height: 226 }, {
    fill: PALETTE.tealPale, stroke: PALETTE.teal, fontSize: 21, name: "user-side-representation",
  });
  const itemBlock = addNode(slide, `${bundle.manifest?.genres?.length ?? 18} movie genre flags\nMulti-hot item vector`, { left: 72, top: 514, width: 245, height: 100 }, {
    fill: PALETTE.orangePale, stroke: PALETTE.orange, fontSize: 21, name: "item-genre-fields",
  });
  const itemEmbed = addNode(slide, "Project the genre vector\nto an 8-dimensional\nitem side vector", { left: 374, top: 510, width: 250, height: 108 }, {
    fill: PALETTE.tealPale, stroke: PALETTE.teal, fontSize: 21, name: "item-side-representation",
  });
  const merge = addNode(slide, "User side vector\n8 values\n\nItem side vector\n8 values", { left: 704, top: 318, width: 226, height: 176 }, {
    fill: PALETTE.greyPale, stroke: PALETTE.blue, fontSize: 21, name: "side-vector-pair",
  });
  const output = addNode(slide, "Outer product\n8 x 8 interaction\nmatrix", { left: 1008, top: 334, width: 204, height: 142 }, {
    fill: PALETTE.bluePale, stroke: PALETTE.blue, fontSize: 22, name: "outer-product-result",
  });
  connect(slide, userBlock, userEmbed, { fromSide: "right", toSide: "left" });
  connect(slide, itemBlock, itemEmbed, { fromSide: "right", toSide: "left" });
  connect(slide, userEmbed, merge, { fromSide: "right", toSide: "left" });
  connect(slide, itemEmbed, merge, { fromSide: "right", toSide: "left" });
  connect(slide, merge, output, { fromSide: "right", toSide: "left", kind: "straight" });
  addText(slide, "ZIP values remain categorical strings so leading zeros are retained. No review text or images enter this implementation.", {
    left: 704, top: 536, width: 500, height: 82,
  }, { fontSize: 20, color: PALETTE.muted }, "feature-note");
  addPaperCitation(slide, "123950-123951", "The paper's attribute-based embedding method is described on printed p.123950. Implementation dimensions and selected MovieLens fields come from docs/methodology.md and data/processed/manifest.json.");
}

function makeSlide6(presentation) {
  const slide = addTitle(presentation, presentation.slides.add(), "The explicit coupling branch", "Outer products expose user-feature by item-feature interactions");
  const user = addNode(slide, "User side vector\n8 values", { left: 60, top: 250, width: 170, height: 86 }, {
    fill: PALETTE.bluePale, stroke: PALETTE.blue, fontSize: 21, name: "explicit-user-vector",
  });
  const item = addNode(slide, "Item side vector\n8 values", { left: 60, top: 416, width: 170, height: 86 }, {
    fill: PALETTE.bluePale, stroke: PALETTE.blue, fontSize: 21, name: "explicit-item-vector",
  });
  const product = addNode(slide, "Outer product\nX = u x v^T\n8 x 8", { left: 286, top: 308, width: 202, height: 126 }, {
    fill: PALETTE.tealPale, stroke: PALETTE.teal, fontSize: 22, name: "explicit-outer-product",
  });
  const conv = addNode(slide, "Implemented local path\nValid 3x3 convolution\n8 output channels\n6 x 6 x 8", { left: 556, top: 204, width: 254, height: 132 }, {
    fill: PALETTE.tealPale, stroke: PALETTE.teal, fontSize: 21, name: "project-convolution",
  });
  const pool = addNode(slide, "ReLU, then 2x2 max pool\n3 x 3 x 8\nFlatten to 72 values", { left: 556, top: 380, width: 254, height: 122 }, {
    fill: PALETTE.tealPale, stroke: PALETTE.teal, fontSize: 21, name: "project-pooling",
  });
  const global = addNode(slide, "Global path\nFlatten original X\n8 x 8 = 64 values", { left: 556, top: 534, width: 254, height: 90 }, {
    fill: PALETTE.orangePale, stroke: PALETTE.orange, fontSize: 20, name: "project-global",
  });
  const combine = addNode(slide, "Explicit representation\n72 local + 64 global\n= 136 values", { left: 922, top: 316, width: 274, height: 154 }, {
    fill: PALETTE.greyPale, stroke: PALETTE.blue, fontSize: 22, name: "explicit-representation",
  });
  connect(slide, user, product, { fromSide: "right", toSide: "left" });
  connect(slide, item, product, { fromSide: "right", toSide: "left" });
  connect(slide, product, conv, { fromSide: "right", toSide: "left" });
  connect(slide, conv, pool, { fromSide: "bottom", toSide: "top", kind: "straight", color: PALETTE.teal });
  connect(slide, product, global, { fromSide: "bottom", toSide: "left", color: PALETTE.orange });
  connect(slide, pool, combine, { fromSide: "right", toSide: "left", color: PALETTE.teal });
  connect(slide, global, combine, { fromSide: "right", toSide: "left", color: PALETTE.orange });
  addText(slide, "Paper ML-1M settings specify two CNN layers. The bounded project model uses one convolution and max pooling, a disclosed computational adaptation.", {
    left: 286, top: 630, width: 910, height: 46,
  }, { fontSize: 18, bold: true, color: PALETTE.red }, "cnn-depth-disclosure");
  addPaperCitation(slide, "123947-123948, 123953", "Figure 1 and Equations 2-3 describe local/global coupling. Printed p.123953 specifies two CNN layers with 3x3 filters and eight channels for MovieLens 1M. The project applies one valid convolution before pooling and does not claim exact architectural replication.");
}

function makeSlide7(presentation) {
  const slide = addTitle(presentation, presentation.slides.add(), "The implemented CoupledCF scorer", "The project compresses each branch before joint fusion");
  const sideInput = addNode(slide, "Attribute pair\n8D user vector\n8D item vector", { left: 54, top: 244, width: 176, height: 118 }, {
    fill: PALETTE.bluePale, stroke: PALETTE.blue, fontSize: 20, name: "implemented-side-input",
  });
  const idInput = addNode(slide, "User ID and\nitem ID", { left: 54, top: 500, width: 176, height: 88 }, {
    fill: PALETTE.orangePale, stroke: PALETTE.orange, fontSize: 21, name: "implemented-id-input",
  });
  const local = addNode(slide, "Local branch\n72 values", { left: 270, top: 190, width: 174, height: 92 }, {
    fill: PALETTE.tealPale, stroke: PALETTE.teal, fontSize: 21, name: "local-72",
  });
  const global = addNode(slide, "Global branch\n64 values", { left: 270, top: 318, width: 174, height: 92 }, {
    fill: PALETTE.orangePale, stroke: PALETTE.orange, fontSize: 21, name: "global-64",
  });
  const implicit = addNode(slide, "32D ID embeddings\nElement product\nDeepCF output 64", { left: 270, top: 490, width: 198, height: 112 }, {
    fill: PALETTE.orangePale, stroke: PALETTE.orange, fontSize: 19, name: "implicit-64",
  });
  const explicitConcat = addNode(slide, "Explicit concat\n72 + 64\n= 136 values", { left: 498, top: 240, width: 170, height: 122 }, {
    fill: PALETTE.greyPale, stroke: PALETTE.blue, fontSize: 20, name: "explicit-concat-136",
  });
  const explicitFusion = addNode(slide, "Explicit MLP\n136 to 64", { left: 710, top: 244, width: 150, height: 112 }, {
    fill: PALETTE.tealPale, stroke: PALETTE.teal, fontSize: 21, name: "explicit-fusion-64",
  });
  const jointConcat = addNode(slide, "Joint concat\n64 + 64\n= 128 values", { left: 906, top: 240, width: 152, height: 122 }, {
    fill: PALETTE.greyPale, stroke: PALETTE.blue, fontSize: 20, name: "joint-concat-128",
  });
  const jointFusion = addNode(slide, "Joint MLP\n128 to 64", { left: 906, top: 464, width: 152, height: 108 }, {
    fill: PALETTE.bluePale, stroke: PALETTE.blue, fontSize: 21, name: "joint-fusion-64",
  });
  const output = addNode(slide, "Scalar logit\nranking score", { left: 1104, top: 464, width: 136, height: 108 }, {
    fill: PALETTE.tealPale, stroke: PALETTE.teal, fontSize: 20, name: "score-output",
  });
  connect(slide, sideInput, local, { fromSide: "right", toSide: "left" });
  connect(slide, sideInput, global, { fromSide: "right", toSide: "left" });
  connect(slide, idInput, implicit, { fromSide: "right", toSide: "left" });
  connect(slide, local, explicitConcat, { fromSide: "right", toSide: "left" });
  connect(slide, global, explicitConcat, { fromSide: "right", toSide: "left" });
  connect(slide, explicitConcat, explicitFusion, { fromSide: "right", toSide: "left", kind: "straight" });
  connect(slide, explicitFusion, jointConcat, { fromSide: "right", toSide: "left", kind: "straight" });
  connect(slide, implicit, jointConcat, { fromSide: "right", toSide: "top" });
  connect(slide, jointConcat, jointFusion, { fromSide: "bottom", toSide: "top", kind: "straight" });
  connect(slide, jointFusion, output, { fromSide: "right", toSide: "left", kind: "straight" });
  addText(slide, "Coupled-local compresses 72 local values to 64, then fuses them with DeepCF's 64 values at width 128.", {
    left: 270, top: 620, width: 616, height: 48,
  }, { fontSize: 18, color: PALETTE.muted }, "coupled-local-shape");
  addText(slide, "Binary cross entropy with logits learns the score from observed interactions and sampled unobserved pairs.", {
    left: 902, top: 602, width: 336, height: 58,
  }, { fontSize: 17, color: PALETTE.muted }, "scorer-loss");
  addPaperCitation(slide, "123948-123950", "Figures 2-3 and Equations 4-16 describe DeepCF and coupled explicit/implicit pathways. In src/coupledcf/models.py, CoupledCF concatenates 72 local and 64 global values, compresses the 136 explicit values to 64, concatenates those 64 with DeepCF's 64-value output to form 128, then applies a joint 128-to-64 MLP and scalar output. Coupled-local compresses 72 local values to 64 before the same joint path. This is the implemented source architecture.");
}

function makeSlide8(presentation, bundle) {
  const slide = addTitle(presentation, presentation.slides.add(), "MovieLens 1M and evaluation protocol", "Fixed user-level ranking candidates support matched model comparisons");
  const manifest = bundle.manifest;
  const ratings = manifest?.ratings?.toLocaleString("en-US") ?? "1,000,209";
  const users = manifest?.users?.toLocaleString("en-US") ?? "6,040";
  const metadataItems = manifest?.metadata_items?.toLocaleString("en-US") ?? "3,883";
  const ratedItems = manifest?.rated_items?.toLocaleString("en-US") ?? "3,706";
  const training = manifest?.train_positives?.toLocaleString("en-US") ?? "988,129";
  const valUsers = manifest?.validation_positives?.toLocaleString("en-US") ?? "6,040";
  const testUsers = manifest?.test_positives?.toLocaleString("en-US") ?? "6,040";
  const candidates = manifest?.candidates_per_user ?? 100;
  const fields = [
    { value: users, label: "users" },
    { value: ratings, label: "observed ratings" },
    { value: `${metadataItems} / ${ratedItems}`, label: "metadata titles / rated titles" },
  ];
  fields.forEach((item, index) => {
    const left = 72 + index * 376;
    addText(slide, item.value, { left, top: 210, width: 330, height: 70 }, {
      fontSize: 40, bold: true, color: [PALETTE.blue, PALETTE.teal, PALETTE.orange][index],
    }, "dataset-stat");
    addText(slide, item.label, { left, top: 284, width: 330, height: 42 }, {
      fontSize: 22, color: PALETTE.muted,
    }, "dataset-stat-label");
  });
  addRule(slide, 72, 354, 1096, PALETTE.line, 1);
  heading(slide, "Training and held-out positives", 72, 384, 470);
  addText(slide, `${training} train positives`, { left: 72, top: 428, width: 360, height: 42 }, {
    fontSize: 25, bold: true, color: PALETTE.ink,
  }, "train-count");
  addText(slide, `${valUsers} validation and ${testUsers} test positives`, { left: 72, top: 476, width: 515, height: 42 }, {
    fontSize: 22, color: PALETTE.body,
  }, "heldout-count");
  addText(slide, "All observed 1-5 ratings count as positives. Four unobserved negatives are sampled per training positive each epoch.", {
    left: 72, top: 534, width: 510, height: 82,
  }, { fontSize: 21 }, "training-sampling");
  heading(slide, "Ranking evaluation", 670, 384, 450);
  addText(slide, `${candidates} candidates per user`, { left: 670, top: 428, width: 440, height: 42 }, {
    fontSize: 25, bold: true, color: PALETTE.ink,
  }, "candidate-count");
  addText(slide, "One held-out positive and 99 items outside the user's observed set. HR@10 and NDCG@10 are primary.", {
    left: 670, top: 476, width: 500, height: 78,
  }, { fontSize: 21 }, "ranking-protocol");
  const seedText = bundle.config?.seeds?.join(", ") ?? "42, 43, 44";
  addText(slide, `Validation is separate from test. Split seed ${manifest?.split_seed ?? 2026}. Model seeds ${seedText}.`, {
    left: 670, top: 572, width: 510, height: 58,
  }, { fontSize: 19, color: PALETTE.muted }, "seed-protocol");
  addPaperCitation(slide, "123951-123953", "The paper treats all observed ratings as positive Top-N interactions, samples four negatives per positive for the implicit task, and evaluates one positive with 99 unobserved candidates using HR and NDCG. The one-per-user validation holdout is a project choice because the paper does not specify its per-user count. Dataset counts come from the verified project manifest.");
}

function makeSlide9(presentation, bundle) {
  const slide = addTitle(presentation, presentation.slides.add(), "Experimental variants and controls", "Every model shares the same split, candidate arrays, seed list and training budget");
  const variants = bundle.config?.variants ?? ["deepcf", "local", "global", "coupled_local", "coupled"];
  const descriptions = {
    deepcf: ["DeepCF", "-", "-", "ID product 32 to 64"],
    local: ["Local CNN", "Local 72 to 64", "-", "64 to score"],
    global: ["Global", "-", "Global 64 to 64", "64 to score"],
    coupled_local: ["Local + implicit", "Local 72 to 64", "-", "Concat 128 to 64"],
    coupled: ["CoupledCF", "Local 72 + global 64 to 64", "Included at left", "Concat 128 to 64"],
  };
  const table = [
    ["Variant", "Explicit branch", "Global path", "Final joint stage"],
    ...variants.map((variant) => descriptions[variant] ?? [variant, "Configured", "Configured", "Configured"]),
  ];
  addNativeTable(slide, table, { left: 72, top: 206, width: 1096, height: 304 }, {
    columnWidths: [220, 350, 250, 326],
    fontSize: 17,
    boldFirstColumn: true,
    rowHeights: [52, 50, 50, 50, 50, 52].slice(0, table.length),
  });
  const train = bundle.config?.training;
  const protocol = train
    ? `${train.epochs} epochs | batch ${train.batch_size} | ${train.device.toUpperCase()} | ${train.threads} threads | lr ${train.learning_rate} | ${train.negatives} train negatives`
    : "10 epochs | batch 4096 | CPU | 2 threads | learning rate 0.001 | 4 train negatives";
  addText(slide, protocol, { left: 72, top: 550, width: 1096, height: 40 }, {
    fontSize: 21, bold: true, color: PALETTE.ink,
  }, "training-budget");
  addText(slide, "Local plus implicit compresses 72 local values to 64 before concatenating with DeepCF's 64-value output. Validation NDCG@10 selects the checkpoint.", {
    left: 72, top: 600, width: 1096, height: 38,
  }, { fontSize: 18, color: PALETTE.muted }, "checkpoint-selection");
  addProjectNotes(slide, ["configs/main.json", "src/coupledcf/models.py", "docs/methodology.md", "results/summary.csv"], "For CoupledCF, local 72 and global 64 values enter an explicit fusion MLP of width 64. That result is concatenated with DeepCF's 64-value representation, then a joint MLP maps 128 to 64 before the output logit. Coupled-local omits the global branch and maps local 72 to explicit 64 before the same joint fusion.");
}

function makeSlide10(presentation, bundle) {
  const slide = addTitle(presentation, presentation.slides.add(), "Top-N model comparison", bundle.resultsReady
    ? "Measured test performance across configured seeds"
    : "The chart and numeric cells remain pending until the complete result bundle validates");
  const variants = bundle.config?.variants ?? ["deepcf", "local", "global", "coupled_local", "coupled"];
  const tableValues = [
    ["Model or reference", "HR@10", "NDCG@10", "Parameters"],
  ];
  if (bundle.resultsReady) {
    for (const variant of variants) {
      const row = bundle.summary.find((item) => item.variant === variant);
      tableValues.push([
        variantLabel(variant),
        formatMetric(row["HR@10_mean"], row["HR@10_std"]),
        formatMetric(row["NDCG@10_mean"], row["NDCG@10_std"]),
        numberFrom(row.parameters, "summary parameter count").toLocaleString("en-US"),
      ]);
    }
    tableValues.push([
      "Train popularity*",
      formatDecimal(bundle.references.popularity_train_only["HR@10"]),
      formatDecimal(bundle.references.popularity_train_only["NDCG@10"]),
      "Reference",
    ]);
    tableValues.push([
      "Random rank**",
      formatDecimal(bundle.references.random_rank_expectation["HR@10"]),
      formatDecimal(bundle.references.random_rank_expectation["NDCG@10"]),
      "Expected",
    ]);
  } else {
    for (const variant of variants) {
      tableValues.push([variantLabel(variant), "Pending", "Pending", "Pending"]);
    }
    tableValues.push(["Train popularity*", "Pending", "Pending", "Reference"]);
    tableValues.push(["Random rank**", "Pending", "Pending", "Expected"]);
  }
  addNativeTable(slide, tableValues, { left: 64, top: 202, width: 634, height: 376 }, {
    columnWidths: [176, 145, 158, 155],
    fontSize: 16,
    boldFirstColumn: true,
    rowHeights: Array(tableValues.length).fill(376 / tableValues.length),
  });
  if (bundle.resultsReady) {
    const categories = variants.map((variant) => variant === "coupled_local" ? "Local+ID" : variantLabel(variant));
    const chart = slide.charts.add("bar", {
      position: { left: 730, top: 198, width: 482, height: 386 },
      title: "Seed means",
      titlePlacement: "aboveChart",
      titleTextStyle: { fontSize: 18, fill: PALETTE.ink, bold: true },
      categories,
      series: [
        {
          name: "HR@10",
          values: variants.map((variant) => chartNumber(csvSummaryValue(bundle.summary.find((row) => row.variant === variant), "HR@10_mean"))),
          fill: PALETTE.teal,
        },
        {
          name: "NDCG@10",
          values: variants.map((variant) => chartNumber(csvSummaryValue(bundle.summary.find((row) => row.variant === variant), "NDCG@10_mean"))),
          fill: PALETTE.orange,
        },
      ],
      barOptions: { direction: "column", grouping: "clustered", gapWidth: 70 },
      hasLegend: true,
      legend: { position: "bottom", overlay: false, textStyle: { fontSize: 16, fill: PALETTE.body } },
      xAxis: { textStyle: { fontSize: 14, fill: PALETTE.body }, majorGridlines: null },
      yAxis: {
        min: 0,
        max: 1,
        majorUnit: 0.2,
        numberFormatCode: "0.0",
        title: { text: "Ranking metric", textStyle: { fontSize: 16, fill: PALETTE.muted } },
        textStyle: { fontSize: 15, fill: PALETTE.muted },
        majorGridlines: { style: "solid", fill: PALETTE.line, width: 1 },
      },
      dataLabels: { showValue: true, position: "outEnd", numberFormatCode: "0.000", textStyle: { fontSize: 12, fill: PALETTE.ink } },
      chartFill: PALETTE.paper,
      chartLine: { style: "solid", fill: "none", width: 0 },
      plotAreaFill: PALETTE.paper,
      plotAreaLine: { style: "solid", fill: "none", width: 0 },
    });
    APPLY_CHART_FONT(chart);
  } else {
    addText(slide, "No partial or pilot metrics appear here. The final build requires every configured model and seed.", {
      left: 748, top: 290, width: 420, height: 148,
    }, { fontSize: 25, bold: true, color: PALETTE.muted, alignment: "center", verticalAlignment: "middle" }, "results-pending");
    addText(slide, "The table will show the five model means and sample SDs plus fixed-split reference baselines.", {
      left: 748, top: 470, width: 420, height: 88,
    }, { fontSize: 20, color: PALETTE.muted, alignment: "center" }, "results-description");
  }
  addText(slide, "* Single fixed-split train-only popularity baseline. ** Uniform-rank expectation. Table entries for trained models show mean +/- sample SD.", {
    left: 64, top: 604, width: 1144, height: 56,
  }, { fontSize: 17, color: PALETTE.muted }, "results-footnote");
  if (bundle.resultsReady) {
    addProjectNotes(slide, ["results/runs.csv", "results/summary.csv", "results/references.json", "data/processed/manifest.json"],
      `The result table and chart were built from a validated ${bundle.runs.length}-run bundle. Split SHA256: ${bundle.references.split_sha256}. Model table values show the arithmetic mean and sample SD across model seeds.`);
  } else {
    addProjectNotes(slide, ["configs/main.json", "results/summary.csv", "results/runs.csv", "results/references.json"],
      "This private audit draft contains status text only. Final numeric results and chart data are omitted until all configured runs and paired checks exist.");
  }
}

function makeSlide11(presentation, bundle) {
  const slide = addTitle(presentation, presentation.slides.add(), "Paired ablation differences", bundle.resultsReady
    ? "Each delta subtracts the matched second model from the first for the same seed"
    : "Four planned comparisons share model seed and evaluation candidates");
  const comparisons = [
    { label: "Full vs DeepCF", key: "add explicit local and global" },
    { label: "Add global path", key: "add global to local plus implicit" },
    { label: "Add local path", key: "add local to implicit" },
    { label: "Local vs global", key: "local-only versus global-only" },
  ];
  const tableValues = [
    ["Paired comparison", "HR@10 delta", "NDCG@10 delta"],
    ...comparisons.map((item) => {
      if (!bundle.resultsReady) return [item.label, "Pending", "Pending"];
      const hr = ablationStats(bundle, item.key, "delta_HR@10");
      const ndcg = ablationStats(bundle, item.key, "delta_NDCG@10");
      return [
        item.label,
        formatMetric(hr.mean, hr.std),
        formatMetric(ndcg.mean, ndcg.std),
      ];
    }),
  ];
  addNativeTable(slide, tableValues, { left: 64, top: 204, width: 608, height: 336 }, {
    columnWidths: [220, 186, 202],
    fontSize: 18,
    boldFirstColumn: true,
    rowHeights: [64, 68, 68, 68, 68],
  });
  if (bundle.resultsReady) {
    const chart = slide.charts.add("bar", {
      position: { left: 710, top: 202, width: 500, height: 342 },
      title: "Mean paired delta",
      titlePlacement: "aboveChart",
      titleTextStyle: { fontSize: 18, fill: PALETTE.ink, bold: true },
      categories: comparisons.map((item) => item.label),
      series: [
        {
          name: "HR@10",
          values: comparisons.map((item) => chartNumber(ablationStats(bundle, item.key, "delta_HR@10").mean)),
          fill: PALETTE.teal,
        },
        {
          name: "NDCG@10",
          values: comparisons.map((item) => chartNumber(ablationStats(bundle, item.key, "delta_NDCG@10").mean)),
          fill: PALETTE.orange,
        },
      ],
      barOptions: { direction: "bar", grouping: "clustered", gapWidth: 55 },
      hasLegend: true,
      legend: { position: "bottom", overlay: false, textStyle: { fontSize: 16, fill: PALETTE.body } },
      xAxis: {
        numberFormatCode: "0.000",
        title: { text: "Metric-point difference", textStyle: { fontSize: 15, fill: PALETTE.muted } },
        textStyle: { fontSize: 14, fill: PALETTE.muted },
        majorGridlines: { style: "solid", fill: PALETTE.line, width: 1 },
      },
      yAxis: { textStyle: { fontSize: 13, fill: PALETTE.body }, majorGridlines: null },
      dataLabels: { showValue: true, position: "outEnd", numberFormatCode: "0.000", textStyle: { fontSize: 11, fill: PALETTE.ink } },
      chartFill: PALETTE.paper,
      chartLine: { style: "solid", fill: "none", width: 0 },
      plotAreaFill: PALETTE.paper,
      plotAreaLine: { style: "solid", fill: "none", width: 0 },
    });
    APPLY_CHART_FONT(chart);
  } else {
    addText(slide, "Paired values will be calculated from ablation.csv and checked against each seed's run records.", {
      left: 730, top: 284, width: 454, height: 148,
    }, { fontSize: 23, bold: true, color: PALETTE.muted, alignment: "center", verticalAlignment: "middle" }, "ablation-pending");
  }
  addText(slide, "Positive values favor the first model. Three seeds describe this fixed split; they do not establish statistical significance.", {
    left: 64, top: 587, width: 1136, height: 54,
  }, { fontSize: 20, color: PALETTE.muted }, "ablation-caution");
  addProjectNotes(slide, ["results/ablation.csv", "results/runs.csv", "configs/main.json"],
    "The CSV's paired differences are re-computed and verified against the same-seed run metrics before use.");
}

function makeSlide12(presentation, bundle) {
  const slide = addTitle(presentation, presentation.slides.add(), "Compute and model cost", "Run time and memory are observed costs on the recorded CPU host");
  const config = bundle.config;
  const training = config?.training;
  const device = training?.device?.toUpperCase() ?? "CPU";
  const epochs = training?.epochs ?? 10;
  const batch = training?.batch_size ?? 4096;
  const threads = training?.threads ?? 2;
  const trainText = `${device} | ${epochs} epochs | batch ${batch} | ${threads} threads`;
  addText(slide, trainText, { left: 64, top: 186, width: 1120, height: 38 }, {
    fontSize: 24, bold: true, color: PALETTE.ink,
  }, "compute-config");
  if (bundle.resultsReady) {
    const values = [
      ["Model", "Parameters", "Wall time / run", "Peak RSS (MiB)"],
      ...config.variants.map((variant) => {
        const row = bundle.summary.find((entry) => entry.variant === variant);
        return [
          variantLabel(variant),
          numberFrom(row.parameters, "model parameter count").toLocaleString("en-US"),
          `${(numberFrom(row.wall_seconds_mean, "wall seconds mean") / 60).toFixed(1)} +/- ${(numberFrom(row.wall_seconds_std, "wall seconds SD") / 60).toFixed(1)} min`,
          `${numberFrom(row.sampled_peak_rss_mb_mean, "RSS mean").toFixed(0)} +/- ${numberFrom(row.sampled_peak_rss_mb_std, "RSS SD").toFixed(0)} MiB`,
        ];
      }),
    ];
    addNativeTable(slide, values, { left: 64, top: 244, width: 708, height: 326 }, {
      columnWidths: [144, 144, 218, 202],
      fontSize: 16,
      boldFirstColumn: true,
      rowHeights: [54, 54, 54, 54, 54, 56],
    });
    const chart = slide.charts.add("bar", {
      position: { left: 804, top: 234, width: 404, height: 346 },
      title: "Mean wall time",
      titlePlacement: "aboveChart",
      titleTextStyle: { fontSize: 18, fill: PALETTE.ink, bold: true },
      categories: config.variants.map(variantLabel),
      series: [{
        name: "Minutes per run",
        values: config.variants.map((variant) =>
          chartNumber(numberFrom(bundle.summary.find((row) => row.variant === variant).wall_seconds_mean, "wall seconds") / 60)),
        fill: PALETTE.blue,
      }],
      barOptions: { direction: "bar", grouping: "clustered", gapWidth: 55 },
      hasLegend: false,
      xAxis: {
        numberFormatCode: "0.0",
        title: { text: "Minutes", textStyle: { fontSize: 15, fill: PALETTE.muted } },
        textStyle: { fontSize: 14, fill: PALETTE.muted },
        majorGridlines: { style: "solid", fill: PALETTE.line, width: 1 },
      },
      yAxis: { textStyle: { fontSize: 14, fill: PALETTE.body }, majorGridlines: null },
      dataLabels: { showValue: true, position: "outEnd", numberFormatCode: "0.0", textStyle: { fontSize: 12, fill: PALETTE.ink } },
      chartFill: PALETTE.paper,
      chartLine: { style: "solid", fill: "none", width: 0 },
      plotAreaFill: PALETTE.paper,
      plotAreaLine: { style: "solid", fill: "none", width: 0 },
    });
    APPLY_CHART_FONT(chart);
  } else {
    addNativeTable(slide, [
      ["Model", "Parameters", "Wall time / run", "Peak RSS (MB)"],
      ...((config?.variants ?? ["deepcf", "local", "global", "coupled_local", "coupled"])
        .map((variant) => [variantLabel(variant), "Pending", "Pending", "Pending"])),
    ], { left: 64, top: 250, width: 708, height: 326 }, {
      columnWidths: [144, 144, 218, 202],
      fontSize: 17,
      boldFirstColumn: true,
    });
    addText(slide, "Costs populate from completed run records.", {
      left: 832, top: 344, width: 340, height: 126,
    }, { fontSize: 24, bold: true, color: PALETTE.muted, alignment: "center", verticalAlignment: "middle" }, "cost-pending");
  }
  addText(slide, "Timing includes sampling, validation, checkpoint writes and final test evaluation. RSS is sampled process memory; neither is a controlled hardware benchmark.", {
    left: 64, top: 596, width: 1144, height: 58,
  }, { fontSize: 18, color: PALETTE.muted }, "cost-caveat");
  addProjectNotes(slide, ["configs/main.json", "results/runs.csv", "results/summary.csv"],
    "Parameter counts come from each instantiated model. Wall time and sampled peak RSS are means and sample SDs across the configured model seeds.");
}

function makeSlide13(presentation, bundle) {
  const slide = addTitle(presentation, presentation.slides.add(), "Published MovieLens 1M reference", "Table 7 reports HR@10 and NDCG@10 for four Top-10 methods");
  const paper = bundle.references?.paper_table7_pdf_page11;
  const table = [
    ["Paper method", "HR@10", "NDCG@10"],
    ...EXPECTED_TABLE7_KEYS.map((variant) => [
      variantLabel(variant),
      paper ? formatDecimal(paper[variant]["HR@10"]) : formatDecimal(PAPER_REFERENCE[variant].hr),
      paper ? formatDecimal(paper[variant]["NDCG@10"]) : formatDecimal(PAPER_REFERENCE[variant].ndcg),
    ]),
  ];
  addNativeTable(slide, table, { left: 72, top: 226, width: 820, height: 304 }, {
    columnWidths: [392, 208, 220],
    fontSize: 23,
    boldFirstColumn: true,
    rowHeights: [60, 61, 61, 61, 61],
  });
  addText(slide, "Paper protocol and implementation details differ from this project. The numbers are context, never targets for tuning.", {
    left: 72, top: 566, width: 1096, height: 68,
  }, { fontSize: 22, bold: true, color: PALETTE.red }, "paper-comparison-caveat");
  addPaperCitation(slide, "123954", "The values are Table 7 for MovieLens 1M. DeepCF: 0.7147/0.4312; local CoupledCF: 0.8212/0.5408; global CoupledCF: 0.7826/0.5252; CoupledCF: 0.8252/0.5440.");
}

function makeSlide14(presentation, bundle) {
  const slide = addTitle(presentation, presentation.slides.add(), "Reading the measured comparison", bundle.resultsReady
    ? "Three seed means on one fixed split provide descriptive evidence"
    : "Conclusions will use the final measured table after every configured run passes validation");
  if (bundle.resultsReady) {
    const coupledHr = metricStatsForVariant(bundle, "coupled", "HR@10");
    const deepHr = metricStatsForVariant(bundle, "deepcf", "HR@10");
    const coupledNdcg = metricStatsForVariant(bundle, "coupled", "NDCG@10");
    const deepNdcg = metricStatsForVariant(bundle, "deepcf", "NDCG@10");
    const hrDiff = coupledHr.mean - deepHr.mean;
    const ndcgDiff = coupledNdcg.mean - deepNdcg.mean;
    const bestHr = [...bundle.summary].sort((a, b) => csvSummaryValue(b, "HR@10_mean") - csvSummaryValue(a, "HR@10_mean"))[0];
    const bestNdcg = [...bundle.summary].sort((a, b) => csvSummaryValue(b, "NDCG@10_mean") - csvSummaryValue(a, "NDCG@10_mean"))[0];
    addPlainColumn(
      slide,
      "CoupledCF versus DeepCF",
      `Paired mean difference: HR@10 ${hrDiff >= 0 ? "+" : ""}${formatDecimal(hrDiff)}; NDCG@10 ${ndcgDiff >= 0 ? "+" : ""}${formatDecimal(ndcgDiff)}.\n\nThese means use matched seeds and identical test candidates.`,
      { left: 72, top: 212, width: 510, height: 324, fontSize: 23 },
      PALETTE.teal,
    );
    addPlainColumn(
      slide,
      "Highest measured means",
      `HR@10: ${variantLabel(bestHr.variant)} at ${formatDecimal(bestHr["HR@10_mean"])}.\n\nNDCG@10: ${variantLabel(bestNdcg.variant)} at ${formatDecimal(bestNdcg["NDCG@10_mean"])}.`,
      { left: 680, top: 212, width: 490, height: 324, fontSize: 24 },
      PALETTE.blue,
    );
    addRule(slide, 72, 572, 1096, PALETTE.line, 1);
    addText(slide, "Three seeds on one random split support a local comparison. They do not establish significance, temporal quality or performance on other datasets.", {
      left: 72, top: 598, width: 1096, height: 58,
    }, { fontSize: 20, bold: true, color: PALETTE.ink }, "result-interpretation-limit");
    addProjectNotes(slide, ["results/summary.csv", "results/runs.csv", "results/ablation.csv"],
      "Each difference uses the three matched seeds. A seed is not an independent user sample or split.");
  } else {
    addText(slide, "No directional conclusion is drawn before the five variants and all configured seeds complete.", {
      left: 112, top: 268, width: 1056, height: 126,
    }, { fontSize: 30, bold: true, color: PALETTE.muted, alignment: "center", verticalAlignment: "middle" }, "interpretation-pending");
    addText(slide, "The final slide text will be generated from summary.csv and verified against the paired run records.", {
      left: 144, top: 430, width: 992, height: 78,
    }, { fontSize: 22, color: PALETTE.muted, alignment: "center" }, "interpretation-pending-details");
    addProjectNotes(slide, ["results/summary.csv", "results/runs.csv"], "Pending full-run evidence. This draft includes no modeled outcome.");
  }
}

function makeSlide15(presentation, bundle) {
  const slide = addTitle(presentation, presentation.slides.add(), "Limitations and next experiment", "The implementation keeps the core outer-product mechanism and discloses its scope");
  addBullets(slide, [
    "The paper uses two CNN layers for ML-1M. This project uses one convolution plus max pooling.",
    "The project uses attributes and genres. Review text, rating prediction and other paper datasets remain outside scope.",
    "Evaluation uses warm users on one random split. It does not test temporal order, cold-start users or transfer.",
    "The project trains for 10 CPU epochs with batch size 4096; the paper reports up to 100 epochs and a batch size of 256.",
    `The paper lists 3,952 items. This project has ${bundle.manifest?.metadata_items?.toLocaleString("en-US") ?? "3,883"} metadata records; the maximum item ID is 3,952.`,
  ], { left: 82, top: 200, width: 1104, height: 310 }, {
    makeNativeBulletParagraphs: MAKE_BULLETS,
    fontSize: 20,
    spaceAfterPoints: 8,
    name: "limitations-list",
  });
  addRule(slide, 82, 532, 1090, PALETTE.teal, 3);
  addText(slide, "Proposed improvement", { left: 82, top: 552, width: 250, height: 36 }, {
    fontSize: 22, bold: true, color: PALETTE.teal,
  }, "improvement-heading");
  addText(slide, "Replace the CNN with field-aware cross attention over demographic fields and genre tokens. Match parameter count, split, seeds and update budget. Improvement remains an untested hypothesis.", {
    left: 340, top: 548, width: 836, height: 72,
  }, { fontSize: 21, bold: true, color: PALETTE.ink }, "improvement-proposal");
  addPaperCitation(slide, "123953-123954", "Printed p.123953 gives the paper's Top-N settings, including two CNN layers, batch size 256, and up to 100 epochs. Dataset scope and adaptations are recorded in docs/methodology.md.");
  addProjectNotes(slide, ["docs/final_report.md", "docs/viva_questions.md"], "The project proposes field-aware cross attention with permutation-invariant genre aggregation to avoid assumed local neighborhoods among arbitrary embedding coordinates. This extension is unimplemented and untested. A parameter-matched comparison with the same split, candidate sets, seeds and optimizer-update budget would test it. Separate full-catalogue and entity-holdout protocols would test generalization.");
}

function safeText(value, fallback) {
  return typeof value === "string" && value.trim() ? value : fallback;
}

function makeSlide16(presentation, bundle, bulletHelper) {
  const slide = addTitle(presentation, presentation.slides.add(), "Project team and viva", "Names and contributions stay editable until the team verifies them");
  const students = Array.isArray(bundle.student?.students) ? bundle.student.students : [];
  const rows = students.length ? students : [
    { name: "[Student name]", roll_number: "[Roll number]", contribution: "[Describe actual individual contribution]" },
    { name: "[Student name]", roll_number: "[Roll number]", contribution: "[Describe actual individual contribution]" },
  ];
  const table = [
    ["Student", "Roll number", "Individual contribution"],
    ...rows.map((student) => [
      safeText(student.name, "[Student name]"),
      safeText(student.roll_number, "[Roll number]"),
      safeText(student.contribution, "[Describe actual individual contribution]"),
    ]),
  ];
  addNativeTable(slide, table, { left: 64, top: 194, width: 1152, height: 204 }, {
    columnWidths: [300, 220, 632],
    fontSize: 18,
    boldFirstColumn: true,
    rowHeights: Array(table.length).fill(204 / table.length),
  });
  heading(slide, "Viva prompts", 64, 438, 300, PALETTE.blue);
  addBullets(slide, [
    "How does the outer product represent a user-item feature interaction?",
    "Why are all observed 1-5 ratings treated as positive labels in this Top-N task?",
    "Which protocol differences limit comparison with the paper's Table 7?",
  ], { left: 64, top: 480, width: 1152, height: 162 }, {
    makeNativeBulletParagraphs: bulletHelper,
    fontSize: 21,
    spaceAfterPoints: 4,
    name: "viva-prompts",
  });
  addProjectNotes(slide, ["configs/student.json", "docs/methodology.md", "sources/assignment.txt"],
    "Confirm each student's name, roll number and actual individual contribution before submission. The prompts are preparation questions, not claims about assigned contributions.");
}

let APPLY_CHART_FONT = () => {};
let MAKE_BULLETS = () => {};

async function renderSlides(presentation, directory) {
  await ensureDirectory(directory);
  const output = [];
  for (let index = 0; index < presentation.slides.items.length; index += 1) {
    const slide = presentation.slides.items[index];
    const filename = `slide-${String(index + 1).padStart(2, "0")}.png`;
    const blob = await presentation.export({ slide, format: "png", scale: 1 });
    const imagePath = path.join(directory, filename);
    await fs.writeFile(imagePath, new Uint8Array(await blob.arrayBuffer()));
    output.push(imagePath);
  }
  return output;
}

async function writeCandidate(PresentationFile, presentation, candidatePath) {
  await ensureDirectory(path.dirname(candidatePath));
  const pptx = await PresentationFile.exportPptx(presentation);
  await pptx.save(candidatePath);
}

async function finalize({ finalizePresentation, candidatePath, finalPath, bundle, fontFamily }) {
  await ensureDirectory(path.dirname(finalPath));
  const finalExists = await exists(finalPath);
  invariant(!finalExists, `Refusing to overwrite existing presentation: ${finalPath}`);
  const receiptPath = path.join(AUDIT_DIR, "receipts", `${path.basename(finalPath, ".pptx")}.validation.json`);
  await ensureDirectory(path.dirname(receiptPath));
  const requiredTables = [9, 10, 11, 12, 13, 16];
  const requiredCharts = MODE === "final" ? [10, 11, 12] : [];
  const result = await finalizePresentation({
    explicitTotalSlideCount: OUTPUT_SLIDE_COUNT,
    requiredNativeTableOwnerSlides: requiredTables,
    requiredNativeChartOwnerSlides: requiredCharts,
    materializeLiteralChartWorkbooks: MODE === "final",
    workspaceDir: ROOT,
    candidatePath,
    finalPath,
    pythonExecutable: RUNTIME_PYTHON,
    integrityValidatorPath: path.join(SKILL_DIR, "container_tools", "inspect_presentation_package_integrity.py"),
    layoutValidatorPath: path.join(SKILL_DIR, "container_tools", "inspect_presentation_layout_geometry.py"),
    layoutArgs: [
      "--expected-slide-size-emu",
      EXPECTED_SLIDE_SIZE_EMU,
      "--expected-slide-count",
      String(OUTPUT_SLIDE_COUNT),
      "--validate-bullet-geometry",
      "--validate-heading-fit",
      "--require-uniform-font-family",
      ...requiredTables.flatMap((number) => ["--require-native-table-slide", String(number)]),
    ],
    fontPolicy: { basis: "design", families: [fontFamily] },
    verifyArtifactToolImport: true,
    receiptPath,
  });
  invariant(
    result.schemaVersion === "presentation-finalization.v1"
      && result.finalSha256
      && result.packageIntegrity?.status === "pass"
      && result.presentationLayout?.exitCode === 0
      && result.firstPartyImport?.passed === true,
    "The presentation finalizer did not return a passing validation receipt.",
  );
  return { ...result, receiptPath, finalPath, resultsReady: bundle.resultsReady };
}

async function main() {
  const bundle = await readBundle();
  const artifact = await loadArtifactTool();
  FONT = artifact.fontFamily;
  APPLY_CHART_FONT = artifact.applyPresentationChartFont;
  MAKE_BULLETS = artifact.makeNativeBulletParagraphs;
  const presentation = artifact.Presentation.create({ slideSize: { width: W, height: H } });
  presentation.title = "CoupledCF on MovieLens 1M";
  presentation.subject = "Partial reproduction of explicit and implicit user-item coupling for Top-N recommendation";

  makeSlide1(presentation, bundle);
  makeSlide2(presentation);
  makeSlide3(presentation);
  makeSlide4(presentation, MAKE_BULLETS);
  makeSlide5(presentation, bundle, MAKE_BULLETS);
  makeSlide6(presentation);
  makeSlide7(presentation);
  makeSlide8(presentation, bundle);
  makeSlide9(presentation, bundle);
  makeSlide10(presentation, bundle);
  makeSlide11(presentation, bundle);
  makeSlide12(presentation, bundle);
  makeSlide13(presentation, bundle);
  makeSlide14(presentation, bundle);
  makeSlide15(presentation, bundle);
  makeSlide16(presentation, bundle, MAKE_BULLETS);
  invariant(presentation.slides.items.length === OUTPUT_SLIDE_COUNT, "Slide count differs from the deck contract.");

  const modeFolder = MODE === "final" ? "final" : "draft";
  const privateRoot = path.join(AUDIT_DIR, modeFolder);
  const candidatePath = path.join(privateRoot, "build", "candidate.pptx");
  const renderDirectory = path.join(privateRoot, "rendered");
  let finalPath;
  if (MODE === "final") {
    finalPath = path.join(ROOT, "deliverables", "CoupledCF_presentation.pptx");
  } else {
    let revision = 1;
    do {
      const suffix = revision === 1 ? "" : `_${String(revision).padStart(2, "0")}`;
      finalPath = path.join(privateRoot, `CoupledCF_presentation_draft_validated${suffix}.pptx`);
      revision += 1;
    } while (await exists(finalPath));
  }
  const rendered = await renderSlides(presentation, renderDirectory);
  await writeCandidate(artifact.PresentationFile, presentation, candidatePath);
  const result = await finalize({
    finalizePresentation: artifact.finalizePresentation,
    candidatePath,
    finalPath,
    bundle,
    fontFamily: artifact.fontFamily,
  });
  const summary = {
    mode: MODE,
    slideCount: presentation.slides.items.length,
    resultBundleStatus: bundle.resultsReady ? "complete_and_validated" : "pending",
    resultsRunCount: bundle.runs?.length ?? 0,
    presentationPath: result.finalPath,
    validationReceipt: result.receiptPath,
    renderedSlideCount: rendered.length,
    renderedDirectory: renderDirectory,
    fontFamily: artifact.fontFamily,
  };
  await fs.writeFile(
    path.join(AUDIT_DIR, modeFolder, "build-summary.json"),
    `${JSON.stringify(summary, null, 2)}\n`,
    "utf8",
  );
  console.log(JSON.stringify(summary, null, 2));
}

main().catch((error) => {
  console.error(error?.stack ?? error);
  process.exitCode = 1;
});
