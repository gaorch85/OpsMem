import argparse
import csv
from collections import Counter
from pathlib import Path

import pandas as pd

from utils.llm import get_current_model_name, llm, parse_json_response


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_NAME = get_current_model_name()
DEFAULT_SELF_CONSISTENCY_ROUNDS = 5
MAX_PARSE_RETRY = 3
REQUIRED_INPUT_COLUMNS = ["prediction", "report", "answer"]
EVAL_COLUMNS = [
    "case_id",
    "score",
    "eval_runs",
    "vote_0",
    "vote_1",
    "vote_2",
    "prediction",
    "report",
    "answer",
]


SYSTEM_PROMPT = """
You are an expert in IT operations, specializing in diagnosing complex system incidents and failures.
You will receive a predicted answer for a single case (containing a final diagnosis and analysis report)
along with the corresponding reference diagnosis (ground-truth root cause).

Score this diagnosis according to the following rules:
2 = The predicted diagnosis exactly matches the reference diagnosis (same root cause);
1 = The predicted diagnosis is a broader issue category that reasonably includes the reference diagnosis;
0 = The predicted diagnosis does not meet the criteria for 1 or 2 (incorrect or irrelevant).

Return only JSON in the following format:
{
  "reasoning": "<brief evaluation rationale>",
  "score": 0
}
"""


def _sanitize_filename(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in name)


DEFAULT_INPUT_PATH = BASE_DIR / "output" / "opsmem" / "answers" / f"{_sanitize_filename(DEFAULT_MODEL_NAME)}.csv"


def _prepare_output_file(output_path: Path) -> list[str]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not output_path.exists():
        pd.DataFrame(columns=EVAL_COLUMNS).to_csv(
            output_path,
            index=False,
            encoding="utf-8",
            quoting=csv.QUOTE_ALL,
            quotechar='"',
            escapechar='"',
            sep=",",
        )
    return EVAL_COLUMNS


def _build_user_prompt(prediction: str, report: str, answer: str, case_id: int) -> str:
    combined_answer = f"{prediction}, {report}".strip(", ").strip()
    return f"""
Case {case_id}:

Predicted answer:
{combined_answer}

Reference diagnosis:
{answer}
"""


def _eval_once(user_prompt: str, model_name: str, temperature: float, max_tokens: int) -> dict:
    last_error = None
    for _ in range(MAX_PARSE_RETRY):
        response = llm(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            model_path=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        try:
            data = parse_json_response(response)
            score = int(data["score"])
            if score not in {0, 1, 2}:
                raise ValueError(f"Invalid score: {score}")
            return {"score": score, "reasoning": data.get("reasoning", "").strip()}
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Failed to parse eval response after {MAX_PARSE_RETRY} attempts: {last_error}")


def _self_consistency_vote(
    user_prompt: str,
    model_name: str,
    temperature: float,
    max_tokens: int,
    min_rounds: int = DEFAULT_SELF_CONSISTENCY_ROUNDS,
) -> tuple[int, Counter, int]:
    """Run repeated LLM judgments and return the unique majority score."""

    if min_rounds < 1:
        raise ValueError("Self-consistency rounds must be at least 1.")

    results = []
    votes = Counter()

    while len(results) < min_rounds:
        result = _eval_once(user_prompt, model_name, temperature, max_tokens)
        results.append(result)
        votes[result["score"]] += 1

    while True:
        max_vote = max(votes.values())
        winners = [score for score, count in votes.items() if count == max_vote]
        if len(winners) == 1:
            return winners[0], votes, len(results)

        result = _eval_once(user_prompt, model_name, temperature, max_tokens)
        results.append(result)
        votes[result["score"]] += 1


def _default_output_path(input_path: Path) -> Path:
    answers_root = BASE_DIR / "output" / "opsmem" / "answers"
    try:
        relative = input_path.relative_to(answers_root)
    except ValueError:
        return input_path.parent.parent / "eval" / input_path.name
    return BASE_DIR / "output" / "opsmem" / "eval" / relative


def _load_input_df(input_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path, encoding="utf-8")
    missing = [column for column in REQUIRED_INPUT_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Input file is missing required columns: {missing}")
    for column in REQUIRED_INPUT_COLUMNS:
        df[column] = df[column].fillna("").astype(str).str.strip()
    return df


def _load_evaluated_case_ids(output_path: Path) -> set[int]:
    if not output_path.exists():
        return set()
    existing_df = pd.read_csv(output_path, encoding="utf-8")
    if "case_id" not in existing_df.columns:
        return set()
    return set(existing_df["case_id"].dropna().astype(int).tolist())


def _append_eval_row(output_path: Path, row: dict) -> None:
    pd.DataFrame([row], columns=EVAL_COLUMNS).to_csv(
        output_path,
        index=False,
        encoding="utf-8",
        quoting=csv.QUOTE_ALL,
        quotechar='"',
        escapechar='"',
        sep=",",
        mode="a",
        header=False,
    )


def main(
    input_path: Path,
    model_name: str,
    temperature: float,
    max_tokens: int,
    output_path: Path | None = None,
    eval_rounds: int = DEFAULT_SELF_CONSISTENCY_ROUNDS,
) -> None:
    output_path = output_path or _default_output_path(input_path)
    _prepare_output_file(output_path)

    case_id = 0
    while True:
        df = _load_input_df(input_path)
        if case_id >= len(df):
            break

        evaluated_case_ids = _load_evaluated_case_ids(output_path)
        if case_id in evaluated_case_ids:
            print(f"[Eval] Case {case_id} already exists in output, skip.")
            case_id += 1
            continue

        row = df.iloc[case_id]
        user_prompt = _build_user_prompt(
            prediction=row["prediction"],
            report=row["report"],
            answer=row["answer"],
            case_id=case_id,
        )

        try:
            final_score, votes, eval_runs = _self_consistency_vote(
                user_prompt=user_prompt,
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                min_rounds=eval_rounds,
            )
            current_row = {
                "case_id": case_id,
                "score": final_score,
                "eval_runs": eval_runs,
                "vote_0": votes.get(0, 0),
                "vote_1": votes.get(1, 0),
                "vote_2": votes.get(2, 0),
                "prediction": row["prediction"],
                "report": row["report"],
                "answer": row["answer"],
            }
            _append_eval_row(output_path, current_row)
            print(f"[Eval] Case {case_id}: score={final_score}, votes={dict(votes)}, runs={eval_runs}")
        except Exception as exc:
            print(f"[Eval] Case {case_id} failed: {exc}")

        case_id += 1

    print(f"[Eval] Completed. Results saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate OpsMem answer CSVs with self-consistent LLM judging.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH, help="Answer CSV to evaluate.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output eval CSV. Default mirrors input under output/opsmem/eval/.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="Model name defined in config.yaml.")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument(
        "--eval-rounds",
        type=int,
        default=DEFAULT_SELF_CONSISTENCY_ROUNDS,
        help="Minimum number of LLM-judge rounds before majority voting. Ties trigger extra rounds.",
    )
    args = parser.parse_args()

    main(
        input_path=args.input,
        model_name=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        output_path=args.output,
        eval_rounds=args.eval_rounds,
    )







