"""Write a maintenance ticket from the command line.

    python cli.py --mock --file samples/synthetic_outer_race.csv
    python cli.py --mock --file data/raw/cwru/237.mat --asset "Pump P-7"
    python cli.py --mock --file data/raw/cmapss/test_FD001.txt --unit 34
    python cli.py --file samples/synthetic_engine.txt        (live: spends API credit)

Sample rate and rpm for files in samples/ come from samples/ground_truth.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pm_agent.agent import TicketAgent
from pm_agent.config import OUT_DIR, SAMPLE_DIR
from pm_agent.llm import LLMClient
from pm_agent.signals import SignalError, load_upload


def sample_metadata(path: Path) -> dict:
    truth = json.loads((SAMPLE_DIR / "ground_truth.json").read_text(encoding="utf-8"))
    entry = truth.get(path.name, {})
    return {k: entry[k] for k in ("fs", "rpm") if k in entry}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--file", required=True, type=Path)
    parser.add_argument("--asset", help="asset name for the ticket")
    parser.add_argument("--fs", type=float, help="sample rate in Hz")
    parser.add_argument("--rpm", type=float, help="shaft speed")
    parser.add_argument("--unit", type=int, help="engine number, for a file holding several engines")
    parser.add_argument("--request", default="Write a maintenance ticket for this signal.")
    parser.add_argument("--mock", action="store_true", help="no API calls")
    parser.add_argument("--provider", help="gemini, anthropic or openai")
    args = parser.parse_args()

    metadata = sample_metadata(args.file)
    try:
        signal = load_upload(args.file.read_bytes(), args.file.name, args.asset, args.fs or metadata.get("fs"),
                             args.rpm or metadata.get("rpm"), args.unit)
    except (SignalError, OSError) as exc:
        raise SystemExit(f"Error: {exc}")

    llm = LLMClient(provider=args.provider, mock=args.mock)
    result = TicketAgent(llm).run(signal, args.request,
                                  on_tool=lambda t: print(f"[tool] {t.name}: {'ok' if t.ok else t.result['error']}"))
    folder = OUT_DIR / signal.signal_id
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "ticket.md").write_text(result.ticket.to_markdown(), encoding="utf-8")
    (folder / "ticket.json").write_text(json.dumps(result.ticket.to_dict(), indent=2), encoding="utf-8")
    print()
    print(result.ticket.to_markdown())
    print(f"Saved to {folder}. Model turns: {result.llm_turns}. Paid API calls: {result.api_calls}.")


if __name__ == "__main__":
    main()
