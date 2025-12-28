from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import AnalysisConfig
from .analyzer import RichnessAnalyzer
from .report import ReportFormatter


def _read_input_text(text_arg: str, *, from_stdin: bool = False, encoding: str = "utf-8") -> str:
    """Resolve input text.

    Priority:
      1) --stdin: read full stdin
      2) If text_arg is a path to an existing file: read file
      3) Otherwise: treat text_arg as raw text
    """
    if from_stdin:
        return sys.stdin.read()

    p = Path(text_arg)
    if p.exists() and p.is_file():
        return p.read_text(encoding=encoding)

    return text_arg


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LexRich semantic richness analyzer")

    # Keep positional for compatibility; now it can be either raw text OR a file path.
    parser.add_argument("text", help="Input text OR a path to a text file")

    parser.add_argument("--config", "-c", help="Path to YAML/JSON analysis config", default=None)
    parser.add_argument("--markdown", action="store_true", help="Output markdown tables instead of JSON")

    # Optional: useful for piping
    parser.add_argument("--stdin", action="store_true", help="Read input text from stdin instead of argument")
    parser.add_argument("--encoding", default="utf-8", help="File encoding when text is a path (default: utf-8)")

    args = parser.parse_args(argv)

    cfg = AnalysisConfig.load(args.config) if args.config else AnalysisConfig()
    analyzer = RichnessAnalyzer(cfg)

    text = _read_input_text(args.text, from_stdin=args.stdin, encoding=args.encoding)

    # Optional sanity check when debugging:
    # if cfg.debug:
    #     print(f"DEBUG: input length={len(text)} head={text[:80].replace(chr(10),'\\\\n')}", file=sys.stderr)

    result = analyzer.analyze(text)
    report = ReportFormatter(result)
    output = report.to_markdown_tables() if args.markdown else report.to_json()
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
