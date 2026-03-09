import ast
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import SecretStr


def create_agent(model: str = "google/gemini-2.0-flash-001") -> ChatOpenAI:
    load_dotenv()
    return ChatOpenAI(
        model=model,
        api_key=SecretStr(os.getenv("OPENROUTER_API_KEY") or ""),
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "http://localhost",
            "X-Title": "Rag",
        },
        temperature=0.7,
    )


def llm_invoke(llm: ChatOpenAI, prompt: str) -> str:
    response = llm.invoke([{"role": "user", "content": prompt}])
    content = response.content
    return content if isinstance(content, str) else json.dumps(content)


def _strip_code_fences(text: str) -> str:
    cleaned = text.strip().strip("'").strip('"').strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        cleaned = cleaned[3:-3].strip()
    for lang in ("json", "python"):
        if cleaned.lower().startswith(lang):
            cleaned = cleaned[len(lang) :].strip()
    return cleaned


def parse_llm_list(llm_output: str) -> list[str]:
    cleaned = _strip_code_fences(llm_output)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(cleaned)
        except (SyntaxError, ValueError):
            print(
                "Warning: Could not parse topic list as JSON/Python. Falling back to comma split."
            )
            return [t.strip() for t in cleaned.split(",") if t.strip()]


def parse_llm_object(llm_output: str) -> dict[str, list[str]]:
    cleaned = _strip_code_fences(llm_output)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        parsed = ast.literal_eval(cleaned)
    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object from LLM response.")
    return parsed


def extract_topics(exam_dir: str, llm: ChatOpenAI) -> list[str]:
    """Pass all exam content to the LLM and get back a master list of topics."""
    prompt_path = Path(__file__).parent / "exams" / "exam_prompt.md"
    prompt_template = prompt_path.read_text(encoding="utf-8")

    joined_exams = ""
    for filename in sorted(os.listdir(exam_dir)):
        if not filename.endswith(".md"):
            continue
        file_path = os.path.join(exam_dir, filename)
        print(f"  Reading: {file_path}")
        joined_exams += Path(file_path).read_text(encoding="utf-8") + "\n\n"

    full_prompt = prompt_template.format(exams=joined_exams)
    raw_response = llm_invoke(llm, full_prompt)
    return parse_llm_list(raw_response)


def classify_exams(
    exam_dir: str, topics: list[str], llm: ChatOpenAI
) -> dict[str, list[str]]:
    """For each exam, ask the LLM which topics from the master list appear in it."""
    classification_prompt = """\
Here is the master list of possible exam topics:
{topics_list}

Please analyze the following exam content. Identify WHICH of the exact topics from the list above appear in this exam as a major part of a question.

IMPORTANT:
1. Only use topics from the provided list. Do not invent new ones.
2. Output a valid JSON object as {{"yyyy-mm-dd": ["Topic A", "Topic B"]}} where the key is the filename stem (without extension) and the value is a list of matching topics.

Exam Content:
{exam_content}
"""

    results: dict[str, list[str]] = {}

    for filename in sorted(os.listdir(exam_dir)):
        if not filename.endswith(".md"):
            continue

        file_path = os.path.join(exam_dir, filename)
        print(f"  Classifying: {filename}")

        content = Path(file_path).read_text(encoding="utf-8")
        full_prompt = classification_prompt.format(
            topics_list=json.dumps(topics),
            exam_content=content,
        )

        raw_response = llm_invoke(llm, full_prompt)
        classification = parse_llm_object(raw_response)

        # The LLM key may be the full filename or the stem; normalise to stem
        stem = Path(filename).stem
        selected = classification.get(stem) or classification.get(filename)
        if selected is None:
            selected = next(iter(classification.values()), [])
        if not isinstance(selected, list):
            selected = [str(selected)]

        results[stem] = [str(t) for t in selected]
        print(f"    -> {len(results[stem])} topics: {results[stem]}")

    return results


def invert_to_topic_map(
    exam_topic_map: dict[str, list[str]], all_topics: list[str]
) -> dict[str, list[str]]:
    """Invert {exam_stem: [topics]} → {topic: [exam_dates]}."""
    # Exam stems are like "2024_08_21"; convert to "2024-08-21"
    result: dict[str, list[str]] = {topic: [] for topic in all_topics}
    for exam_stem, topics in exam_topic_map.items():
        date = exam_stem.replace("_", "-")
        for topic in topics:
            if topic in result:
                result[topic].append(date)
    return result


def sort_by_frequency(topic_map: dict[str, list[str]]) -> dict[str, int]:
    """Sort a topic → exam-dates mapping by descending frequency, returning topic → count."""
    return dict(
        sorted(
            ((topic, len(dates)) for topic, dates in topic_map.items()),
            key=lambda item: item[1],
            reverse=True,
        )
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze past exams and produce a topic → exam-dates mapping."
    )
    parser.add_argument(
        "exam_dir",
        type=Path,
        nargs="?",
        help="Directory containing exam .md files (e.g. courses/FMSF20/exams)",
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        default=None,
        help="Existing JSON file (output format) to sort by frequency instead of running LLM analysis",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write JSON result to this file instead of stdout",
    )
    parser.add_argument(
        "--model",
        default="arcee-ai/trinity-large-preview:free",
        help="OpenRouter model to use",
    )
    args = parser.parse_args()

    if args.input_json:
        topic_to_exams = json.loads(args.input_json.read_text(encoding="utf-8"))
    else:
        if not args.exam_dir:
            parser.error("exam_dir is required when --input-json is not provided")

        llm = create_agent(model=args.model)

        print("--- STEP 1: Extracting global topics from all exams ---")
        topics = extract_topics(str(args.exam_dir), llm)
        print(f"\nIdentified {len(topics)} topics: {topics}\n")

        print("--- STEP 2: Classifying each exam against the topic list ---")
        exam_topic_map = classify_exams(str(args.exam_dir), topics, llm)

        print("\n--- STEP 3: Building topic → exam-dates mapping ---")
        topic_to_exams = invert_to_topic_map(exam_topic_map, topics)

    topic_to_exams = sort_by_frequency(topic_to_exams)

    output_json = json.dumps(topic_to_exams, indent=2)

    if args.output:
        args.output.write_text(output_json, encoding="utf-8")
        print(f"\nResults saved to: {args.output}")
    else:
        print("\n--- FINAL RESULTS ---")
        print(output_json)
