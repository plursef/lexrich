from __future__ import annotations

import argparse
import sys

from .config import AnalysisConfig
from .analyzer import RichnessAnalyzer
from .report import ReportFormatter


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LexRich semantic richness analyzer")
    parser.add_argument("text", help="Input text to analyze")
    parser.add_argument("--config", "-c", help="Path to YAML/JSON config", default=None)
    parser.add_argument("--markdown", action="store_true", help="Output markdown tables instead of JSON")
    args = parser.parse_args(argv)

    cfg = AnalysisConfig.load(args.config) if args.config else AnalysisConfig()
    analyzer = RichnessAnalyzer(cfg)
    result = analyzer.analyze(args.text)
    report = ReportFormatter(result)
    output = report.to_markdown_tables() if args.markdown else report.to_json()
    print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
