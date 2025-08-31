#!/usr/bin/env python3
"""
Update tag alias dictionary in app/main.py from tag/config definitions.

This script merges existing manual aliases with aliases from:
- tag/config/tag_definitions_enhanced.json (preferred)
- tag/config/tag_definitions.json (fallback)
- tag/config/tag_definition.json (fallback if present)

It replaces the dictionary between markers in app/main.py:
    # BEGIN_TAG_ALIASES (auto-generated; do not edit by hand)
    tag_aliases = { ... }
    # END_TAG_ALIASES

Run:
    python scripts/update_tag_aliases.py
"""

import ast
import io
import json
import os
import re
from typing import Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_MAIN = os.path.join(ROOT, 'app', 'main.py')
CFG_DIR = os.path.join(ROOT, 'tag', 'config')

BEGIN_MARK = '# BEGIN_TAG_ALIASES (auto-generated; do not edit by hand)'
END_MARK = '# END_TAG_ALIASES'


def load_current_alias_block(text: str) -> Dict[str, List[str]]:
    """Extract and parse the current tag_aliases dict from main.py text."""
    begin_idx = text.find(BEGIN_MARK)
    end_idx = text.find(END_MARK)
    if begin_idx == -1 or end_idx == -1:
        raise RuntimeError('Could not find alias markers in app/main.py')

    # Extract the block
    block = text[begin_idx:end_idx].splitlines()
    # Find the line with 'tag_aliases = {' and capture until the matching closing '}'
    start_line = None
    for i, line in enumerate(block):
        if 'tag_aliases' in line and '=' in line and '{' in line:
            start_line = i
            break
    if start_line is None:
        raise RuntimeError('Could not find tag_aliases dict start in the block')

    # Collect lines from start_line to end of block
    dict_lines = []
    brace = 0
    started = False
    for line in block[start_line:]:
        if '{' in line:
            brace += line.count('{')
            started = True
        if started:
            dict_lines.append(line)
        if '}' in line:
            brace -= line.count('}')
            if started and brace <= 0:
                break

    dict_text = '\n'.join(dict_lines)
    # Build a minimal Python snippet to eval the dict safely
    snippet = dict_text
    # Extract the RHS only
    m = re.search(r'tag_aliases\s*=\s*({[\s\S]*})', snippet)
    if not m:
        raise RuntimeError('Failed to capture dict literal for tag_aliases')
    rhs = m.group(1)
    try:
        current = ast.literal_eval(rhs)
    except Exception as e:
        raise RuntimeError(f'Failed to parse existing alias dict: {e}')
    if not isinstance(current, dict):
        raise RuntimeError('Parsed alias block is not a dict')
    return {str(k): list(v) for k, v in current.items()}


def load_config_aliases() -> Dict[str, List[str]]:
    """Load aliases from config JSON files."""
    candidates = [
        os.path.join(CFG_DIR, 'tag_definitions_enhanced.json'),
        os.path.join(CFG_DIR, 'tag_definitions.json'),
        os.path.join(CFG_DIR, 'tag_definition.json'),
    ]
    for path in candidates:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            aliases: Dict[str, List[str]] = {}
            for entry in data.get('tags', []):
                name = entry.get('name')
                vals = entry.get('aliases') or []
                if name and isinstance(vals, list):
                    aliases[name] = [str(v) for v in vals if isinstance(v, str)]
            return aliases
    return {}


def merge_aliases(base: Dict[str, List[str]], extra: Dict[str, List[str]]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {k: list(v) for k, v in base.items()}
    for name, vals in extra.items():
        lst = out.setdefault(name, [])
        seen = set(lst)
        for v in vals:
            if v not in seen:
                lst.append(v)
                seen.add(v)
    return out


def render_alias_block(aliases: Dict[str, List[str]], indent: str) -> str:
    keys = sorted(aliases.keys())
    buf = io.StringIO()
    buf.write(f"{indent}{BEGIN_MARK}\n")
    buf.write(f"{indent}tag_aliases = {{\n")
    for k in keys:
        vs = aliases[k]
        # Preserve order, quote strings safely
        vs_escaped = ", ".join([repr(x) for x in vs])
        buf.write(f"{indent}    {repr(k)}: [{vs_escaped}],\n")
    buf.write(f"{indent}}}\n")
    buf.write(f"{indent}{END_MARK}\n")
    return buf.getvalue()


def main() -> None:
    with open(APP_MAIN, 'r', encoding='utf-8') as f:
        text = f.read()

    current = load_current_alias_block(text)
    extra = load_config_aliases()
    merged = merge_aliases(current, extra)

    # Detect indentation from the current block start line
    line_before = None
    for line in text.splitlines():
        if BEGIN_MARK in line:
            line_before = line
            break
    if line_before is None:
        raise RuntimeError('Could not detect alias block start line for indentation')
    indent = re.match(r"^\s*", line_before).group(0)

    new_block = render_alias_block(merged, indent)

    # Replace the block between markers
    pattern = re.compile(
        rf"{re.escape(BEGIN_MARK)}[\s\S]*?{re.escape(END_MARK)}",
        re.MULTILINE,
    )
    new_text = pattern.sub(new_block.strip(), text)

    with open(APP_MAIN, 'w', encoding='utf-8') as f:
        f.write(new_text)

    print(f"✅ Updated tag aliases in {APP_MAIN} (total {len(merged)} entries)")


if __name__ == '__main__':
    main()

