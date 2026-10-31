#!/usr/bin/env python3
"""
main.py
-------
CLI entry point for the RAG system.

Usage:
  python main.py ingest              # Load, chunk, embed, store documents
  python main.py ingest --reset      # Wipe store, then ingest
  python main.py query               # Interactive query loop
  python main.py query "How do I reset my password?"   # Single query
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# ── Load .env before importing anything that reads env vars ─────────────────
from dotenv import load_dotenv

# Look for .env in the same directory as this file
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    load_dotenv(dotenv_path=_env_path)
    print(f"[Config] Loaded environment from {_env_path}")
else:
    env_example = Path(__file__).parent / ".env.example"
    print(
        f"[Warning] No .env file found at {_env_path}\n"
        f"          Copy .env.example → .env and fill in your values:\n"
        f"          cp {env_example} {_env_path}"
    )

from app import get_logger
from app.pipeline import IngestionPipeline, QueryPipeline

logger = get_logger(__name__)

# ── Parsers ───────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RAG System — Retrieval-Augmented Generation knowledge assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py ingest\n"
            "  python main.py ingest --reset\n"
            "  python main.py query\n"
            '  python main.py query "How do I reset my password?"\n'
        ),
    )
    subparsers = parser.add_subparsers(dest="command", help="Sub-command")

    # ── ingest ───────────────────────────────────────────────────────────────
    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Load documents, embed them, and store in ChromaDB.",
    )
    ingest_parser.add_argument(
        "--data-dir",
        default=None,
        help="Path to document directory (overrides DATA_DIR in .env).",
    )
    ingest_parser.add_argument(
        "--reset",
        action="store_true",
        help="Wipe the ChromaDB collection before ingesting (full rebuild).",
    )

    # ── query ────────────────────────────────────────────────────────────────
    query_parser = subparsers.add_parser(
        "query",
        help="Ask a question against the stored knowledge base.",
    )
    query_parser.add_argument(
        "question",
        nargs="?",
        default=None,
        help="The question to ask.  Omit for interactive mode.",
    )

    return parser


# ── Handlers ──────────────────────────────────────────────────────────────────

def handle_ingest(args: argparse.Namespace) -> int:
    """Run the ingestion pipeline."""
    data_dir = args.data_dir or os.getenv("DATA_DIR", "./data")
    print(f"\n🗄  Ingestion Pipeline")
    print(f"   Data directory : {data_dir}")
    print(f"   Reset store    : {args.reset}\n")

    try:
        pipeline = IngestionPipeline(data_dir=data_dir)
        summary = pipeline.run(reset=args.reset)
    except FileNotFoundError as exc:
        print(f"\n❌  Error: {exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(f"\n❌  Error: {exc}", file=sys.stderr)
        return 1

    print(f"\n✅  Ingestion complete:")
    print(f"   Documents loaded : {summary['documents']}")
    print(f"   Chunks created   : {summary['chunks']}")
    print(f"   Chunks in store  : {summary['stored']}")
    return 0


def handle_query(args: argparse.Namespace) -> int:
    """Run the query pipeline (single-shot or interactive loop)."""
    try:
        pipeline = QueryPipeline()
    except Exception as exc:  # noqa: BLE001
        print(f"\n❌  Failed to initialise query pipeline: {exc}", file=sys.stderr)
        return 1

    if args.question:
        # Single-shot mode
        _ask_and_print(pipeline, args.question)
        return 0

    # Interactive loop
    print("\n🤖  RAG Knowledge Assistant — Interactive Mode")
    print("   Type your question and press Enter.")
    print("   Type 'exit' or 'quit' to stop.\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋  Goodbye!")
            break

        if question.lower() in {"exit", "quit", "q", "bye"}:
            print("👋  Goodbye!")
            break
        if not question:
            continue

        _ask_and_print(pipeline, question)

    return 0


def _ask_and_print(pipeline: QueryPipeline, question: str) -> None:
    """Run one query and print the answer."""
    print(f"\n❓  {question}\n")
    try:
        answer = pipeline.run(question)
    except Exception as exc:  # noqa: BLE001
        print(f"❌  Query failed: {exc}", file=sys.stderr)
        return
    print("🤖  Answer:\n")
    print(answer)
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    if args.command == "ingest":
        return handle_ingest(args)
    if args.command == "query":
        return handle_query(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
