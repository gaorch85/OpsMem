import argparse
from pathlib import Path

from pipeline import OpsMemPipeline


BASE_DIR = Path(__file__).resolve().parent


def main(
    model_name: str | None = None,
    start_case: int | None = None,
    consolidation_enabled: bool | None = None,
    output_experiment: str | None = None,
    output_answer_subdir: str | None = None,
    case_pause_seconds: float | None = None,
) -> None:
    pipeline = OpsMemPipeline(
        BASE_DIR,
        model_name=model_name,
        consolidation_enabled=consolidation_enabled,
        output_experiment=output_experiment,
        output_answer_subdir=output_answer_subdir,
        case_pause_seconds=case_pause_seconds,
    )
    pipeline.run(start_case=start_case)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the OpsMem diagnosis pipeline.")
    parser.add_argument("--model", dest="model_name", default=None, help="Model name defined in config.yaml.")
    parser.add_argument("--start-case", type=int, default=None, help="Start index of cases. Default: read from code/start_case.tmp.")
    parser.add_argument("--output-experiment", default=None, help="Output subdirectory under output/. Default: opsmem.")
    parser.add_argument("--output-answer-subdir", default=None, help="Optional subdirectory under output/<experiment>/answers/.")
    parser.add_argument("--case-pause-seconds", type=float, default=None, help="Optional pause before each case. Default: 0.")
    parser.add_argument(
        "--consolidation",
        dest="consolidation_enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override LTM consolidation for this run. Use --no-consolidation to disable memory consolidation.",
    )
    args = parser.parse_args()
    main(
        model_name=args.model_name,
        start_case=args.start_case,
        consolidation_enabled=args.consolidation_enabled,
        output_experiment=args.output_experiment,
        output_answer_subdir=args.output_answer_subdir,
        case_pause_seconds=args.case_pause_seconds,
    )






