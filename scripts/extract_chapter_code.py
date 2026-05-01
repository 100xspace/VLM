#!/usr/bin/env python3
"""Extract chapter-wise code snippets from book .docx files."""

from __future__ import annotations

import json
import re
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable
from zipfile import ZipFile

NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "vlm-chapter-code" / "chapters"

HEADING_STYLES = {
    "ChapterTitleNumberBPBHEB",
    "ChapterTitleBPBHEB",
    "Heading1BPBHEB",
    "Heading2BPBHEB",
    "Heading3BPBHEB",
}
CODE_STYLE = "CodeBlockBPBHEB"

STRONG_CODE_PATTERNS = [
    r"^(import|from)\s+\w",
    r"^(def|class)\s+\w",
    r"^(if|elif|else:|for|while|try:|except|with)\b",
    r"^(return|yield|raise|assert)\b",
    r"^(@\w+)",
    r"^(pip install|python |python3 |git clone|curl |wget |cd |ls\b|mkdir\b|export\b)",
    r"^(!pip|!kaggle|!python|!git|!wget|!curl)",
    r"^(model|processor|tokenizer|dataset|train_loader|val_loader|test_loader|results|image|inputs)\s*=",
    r"^(<\?xml|<!DOCTYPE|<html|version:|services:)",
    r"^(MODEL_NAME|DEVICE|BATCH_SIZE|LEARNING_RATE|EPOCHS)\s*=",
]

CONTINUATION_PATTERNS = [
    r"^[\]\)\}\,]$",
    r"^[\]\)\}\,\:]",
    r"^[A-Za-z_][A-Za-z0-9_]*\s*=",
    r"^\s*[A-Za-z_][A-Za-z0-9_]*\s*:",
    r'^\s*"""',
    r"^\s*#",
    r"^\s*$",
]

SHELL_HINTS = ("!pip", "!kaggle", "pip install", "git clone", "curl ", "wget ", "python ", "python3 ")

IMPORT_TO_PACKAGE = {
    "PIL": "Pillow",
    "cv2": "opencv-python",
    "datasets": "datasets",
    "flash_attn": "flash-attn",
    "matplotlib": "matplotlib",
    "nltk": "nltk",
    "numpy": "numpy",
    "onnx": "onnx",
    "onnxoptimizer": "onnxoptimizer",
    "onnxruntime": "onnxruntime",
    "pandas": "pandas",
    "peft": "peft",
    "pydicom": "pydicom",
    "requests": "requests",
    "seaborn": "seaborn",
    "skimage": "scikit-image",
    "sklearn": "scikit-learn",
    "streamlit": "streamlit",
    "torch": "torch",
    "torchvision": "torchvision",
    "transformers": "transformers",
    "trl": "trl",
    "ultralytics": "ultralytics",
    "vllm": "vllm",
}


@dataclass
class Paragraph:
    text: str
    style: str | None


@dataclass
class CodeBlock:
    heading: str
    source_docx: str
    chapter_number: int
    block_index: int
    lines: list[str] = field(default_factory=list)
    script_hint: str | None = None
    extension: str = ".py"
    relative_path: str = ""

    @property
    def line_count(self) -> int:
        return len([line for line in self.lines if line.strip()])


def slugify(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = value.strip("-")
    return value or "snippet"


def paragraph_text(paragraph: ET.Element) -> str:
    parts = [node.text or "" for node in paragraph.findall(".//w:t", NS)]
    return "".join(parts).replace("\xa0", " ").rstrip()


def paragraph_style(paragraph: ET.Element) -> str | None:
    style = paragraph.find("./w:pPr/w:pStyle", NS)
    if style is None:
        return None
    return style.attrib.get(f"{{{NS['w']}}}val")


def load_paragraphs(docx_path: Path) -> list[Paragraph]:
    with ZipFile(docx_path) as archive:
        root = ET.fromstring(archive.read("word/document.xml"))
    paragraphs = []
    for paragraph in root.findall(".//w:p", NS):
        paragraphs.append(Paragraph(text=paragraph_text(paragraph), style=paragraph_style(paragraph)))
    return paragraphs


def extract_title(paragraphs: Iterable[Paragraph], source_name: str) -> tuple[int, str]:
    chapter_number = 0
    title = ""
    for paragraph in paragraphs:
        text = paragraph.text.strip()
        if paragraph.style == "ChapterTitleNumberBPBHEB":
            match = re.search(r"Chapter\s+(\d+)", text, re.IGNORECASE)
            if match:
                chapter_number = int(match.group(1))
        elif paragraph.style == "ChapterTitleBPBHEB" and text:
            title = text
            break
    if chapter_number == 0:
        match = re.search(r"[Cc]hapter[_\s](\d+)", source_name)
        if match:
            chapter_number = int(match.group(1))
    return chapter_number, title or "Untitled Chapter"


def is_strong_code_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if any(re.match(pattern, stripped) for pattern in STRONG_CODE_PATTERNS):
        return True
    if stripped.startswith("#") and any(token in stripped.lower() for token in ("example", "step", "script")):
        return True
    if re.search(r"\w+\([^)]*\)", stripped) and any(op in stripped for op in ("=", ".", ":", ",")):
        return True
    return False


def is_code_continuation(line: str) -> bool:
    stripped = line.strip()
    if any(re.match(pattern, stripped) for pattern in CONTINUATION_PATTERNS):
        return True
    if stripped.endswith(("{", "[", "(", ",", ":", "\\")):
        return True
    if any(token in stripped for token in ("torch.", "np.", "cv2.", "plt.", "Image.", "requests.", "YOLO(", "Auto")):
        return True
    if re.search(r"[{}\[\]()=:]", stripped) and len(stripped.split()) <= 12:
        return True
    return False


def looks_like_prose(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if stripped.endswith(".") and len(stripped.split()) > 10 and "=" not in stripped and "(" not in stripped:
        return True
    if stripped.startswith(("The ", "This ", "These ", "In ", "By ", "With ", "Once ", "After ")):
        return True
    return False


def infer_extension(lines: list[str], script_hint: str | None) -> str:
    if script_hint:
        return Path(script_hint).suffix or ".py"
    first_non_empty = next((line.strip() for line in lines if line.strip()), "")
    if first_non_empty.startswith(SHELL_HINTS):
        return ".sh"
    if first_non_empty.startswith("{") or first_non_empty.startswith("["):
        return ".json"
    if first_non_empty.startswith(("version:", "services:")):
        return ".yaml"
    if first_non_empty.startswith(("<?xml", "<!DOCTYPE", "<html")):
        return ".html"
    return ".py"


def clean_lines(lines: list[str], extension: str) -> list[str]:
    cleaned = [line.replace("\u200b", "").rstrip() for line in lines]
    if extension == ".sh":
        cleaned = [line[1:] if line.startswith("!") else line for line in cleaned]
        if cleaned and not cleaned[0].startswith("#!/"):
            cleaned.insert(0, "#!/usr/bin/env bash")
            cleaned.insert(1, "set -euo pipefail")
            cleaned.insert(2, "")
    return cleaned


def build_blocks(docx_path: Path) -> tuple[int, str, list[CodeBlock]]:
    paragraphs = load_paragraphs(docx_path)
    chapter_number, chapter_title = extract_title(paragraphs, docx_path.name)
    blocks: list[CodeBlock] = []
    current_lines: list[str] = []
    current_heading = chapter_title
    current_script_hint: str | None = None
    started = False
    last_meaningful_line = ""

    for index, paragraph in enumerate(paragraphs):
        text = paragraph.text
        stripped = text.strip()
        style = paragraph.style

        if style in HEADING_STYLES and stripped:
            current_heading = stripped

        if stripped.startswith("Script:"):
            current_script_hint = stripped.split(":", 1)[1].strip()
            continue

        explicit_code = style == CODE_STYLE
        strong = is_strong_code_line(stripped)
        continuation = is_code_continuation(stripped)

        if explicit_code or strong or (started and (continuation or not stripped)):
            if stripped or current_lines:
                current_lines.append(text.replace("\xa0", " "))
            started = True
            if stripped:
                last_meaningful_line = stripped
            continue

        should_flush = started and current_lines
        if should_flush:
            extension = infer_extension(current_lines, current_script_hint)
            cleaned = clean_lines(current_lines, extension)
            non_empty = [line for line in cleaned if line.strip()]
            strong_lines = sum(1 for line in non_empty if is_strong_code_line(line))
            if non_empty and strong_lines >= 1 and not all(looks_like_prose(line) for line in non_empty):
                block = CodeBlock(
                    heading=current_heading,
                    source_docx=docx_path.name,
                    chapter_number=chapter_number,
                    block_index=len(blocks) + 1,
                    lines=cleaned,
                    script_hint=current_script_hint,
                    extension=extension,
                )
                blocks.append(block)
            current_lines = []
            started = False
            current_script_hint = None
            last_meaningful_line = ""

        if stripped and not continuation:
            current_script_hint = None

    if started and current_lines:
        extension = infer_extension(current_lines, current_script_hint)
        cleaned = clean_lines(current_lines, extension)
        non_empty = [line for line in cleaned if line.strip()]
        strong_lines = sum(1 for line in non_empty if is_strong_code_line(line))
        if non_empty and strong_lines >= 1 and not all(looks_like_prose(line) for line in non_empty):
            blocks.append(
                CodeBlock(
                    heading=current_heading,
                    source_docx=docx_path.name,
                    chapter_number=chapter_number,
                    block_index=len(blocks) + 1,
                    lines=cleaned,
                    script_hint=current_script_hint,
                    extension=extension,
                )
            )

    return chapter_number, chapter_title, blocks


def assign_relative_path(block: CodeBlock, used_paths: set[str]) -> None:
    if block.script_hint:
        candidate = block.script_hint.replace("\\", "/").lstrip("./")
        candidate = re.sub(r"\s+", "_", candidate)
        if not Path(candidate).suffix:
            candidate = f"{candidate}{block.extension}"
    else:
        filename = f"{block.block_index:02d}_{slugify(block.heading)[:50]}{block.extension}"
        candidate = f"snippets/{filename}"

    path = Path(candidate)
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    counter = 2
    unique = path.as_posix()
    while unique in used_paths:
        unique = (parent / f"{stem}_{counter}{suffix}").as_posix()
        counter += 1
    used_paths.add(unique)
    block.relative_path = unique


def extract_imports(lines: Iterable[str]) -> set[str]:
    imports: set[str] = set()
    pattern = re.compile(r"^(?:from|import)\s+([A-Za-z0-9_\.]+)")
    for line in lines:
        match = pattern.match(line.strip())
        if not match:
            continue
        module = match.group(1).split(".")[0]
        package = IMPORT_TO_PACKAGE.get(module)
        if package:
            imports.add(package)
    return imports


def write_chapter(chapter_dir: Path, chapter_number: int, chapter_title: str, source_docx: str, blocks: list[CodeBlock]) -> dict:
    chapter_dir.mkdir(parents=True, exist_ok=True)
    code_files = []
    used_paths: set[str] = set()
    dependencies: set[str] = set()

    for block in blocks:
        assign_relative_path(block, used_paths)
        target = chapter_dir / block.relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        content = "\n".join(block.lines).strip() + "\n"
        target.write_text(content, encoding="utf-8")
        code_files.append(
            {
                "heading": block.heading,
                "path": block.relative_path,
                "line_count": block.line_count,
                "source_docx": block.source_docx,
            }
        )
        dependencies.update(extract_imports(block.lines))

    readme_lines = [
        f"# Chapter {chapter_number}: {chapter_title}",
        "",
        f"- Source: `{source_docx}`",
        f"- Extracted code files: `{len(code_files)}`",
        "",
    ]
    if code_files:
        readme_lines.append("## Files")
        readme_lines.append("")
        for item in code_files:
            readme_lines.append(f"- `{item['path']}` from section `{item['heading']}`")
    else:
        readme_lines.extend(
            [
                "## Notes",
                "",
                "No executable code snippets were confidently detected in this chapter.",
            ]
        )
    readme_lines.append("")
    (chapter_dir / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")

    return {
        "chapter_number": chapter_number,
        "chapter_title": chapter_title,
        "source_docx": source_docx,
        "code_files": code_files,
        "dependencies": sorted(dependencies),
    }


def main() -> None:
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    manifests = []
    all_dependencies: set[str] = set()
    manifests_unsorted = []
    for docx_path in sorted(ROOT.glob("*.docx")):
        chapter_number, chapter_title, blocks = build_blocks(docx_path)
        chapter_slug = slugify(chapter_title)[:60]
        chapter_dir = OUTPUT_ROOT / f"chapter_{chapter_number:02d}_{chapter_slug}"
        manifest = write_chapter(chapter_dir, chapter_number, chapter_title, docx_path.name, blocks)
        manifests_unsorted.append(manifest)
        all_dependencies.update(manifest["dependencies"])

    manifests = sorted(manifests_unsorted, key=lambda item: item["chapter_number"])

    manifest_path = ROOT / "vlm-chapter-code" / "manifest.json"
    manifest_path.write_text(json.dumps(manifests, indent=2), encoding="utf-8")

    requirements = [
        "# Common libraries referenced by extracted chapter code",
        "numpy",
        "torch",
        "torchvision",
        "transformers",
        "matplotlib",
        "Pillow",
        "opencv-python",
        "pandas",
        "requests",
        "scikit-image",
        "scikit-learn",
        "pydicom",
        "streamlit",
        "ultralytics",
        "datasets",
        "peft",
        "trl",
        "onnx",
        "onnxruntime",
        "onnxoptimizer",
        "vllm",
        "nltk",
    ]
    req_path = ROOT / "vlm-chapter-code" / "requirements.txt"
    req_path.write_text("\n".join(requirements) + "\n", encoding="utf-8")

    summary_lines = [
        "# VLM Chapter Code",
        "",
        "This repository contains chapter-wise code extracted from the book manuscripts in this workspace.",
        "",
        "## Regenerate",
        "",
        "```bash",
        "python3 scripts/extract_chapter_code.py",
        "```",
        "",
        "## Structure",
        "",
    ]
    for manifest in manifests:
        summary_lines.append(
            f"- `chapters/chapter_{manifest['chapter_number']:02d}_{slugify(manifest['chapter_title'])[:60]}`"
            f" with `{len(manifest['code_files'])}` extracted files"
        )
    summary_lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Some chapters are conceptual and do not contain runnable code snippets.",
            "- A few snippets may still need light cleanup because the manuscript formatting is not fully uniform.",
            "- `requirements.txt` is a practical starting point compiled from imported libraries across the extracted snippets.",
        ]
    )
    (ROOT / "vlm-chapter-code" / "README.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"Extracted {sum(len(item['code_files']) for item in manifests)} files across {len(manifests)} chapters.")


if __name__ == "__main__":
    main()
