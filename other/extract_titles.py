import os
import json
from pathlib import Path
from collections import defaultdict

def extract_titles_from_files(base_path: str) -> dict[str, list[str]]:
    # Initialize storage
    difficulty_titles = defaultdict(list)

    # Define expected difficulties
    expected_difficulties = ["beginner", "elementary", "intermediate", "upper-intermediate", "advanced"]

    # Loop through files
    for difficulty in expected_difficulties:
        file_path = Path(base_path) / f"scenarios_{difficulty}.json"
        if not file_path.exists():
            print(f"Warning: {file_path} not found.")
            continue

        try:
            with file_path.open(encoding="utf-8") as f:
                data = json.load(f)

            # Get the first-level keys (titles)
            titles = list(data.keys())
            difficulty_titles[difficulty] = titles

            # append an index number 'N.' to all titles
            difficulty_titles[difficulty] = [f"{i + 1}. {title}" for i, title in enumerate(titles)]

        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    return difficulty_titles


def main():
    base_path = Path(__file__).resolve().parents[1] / "tutors" / "tutor2" / "scenarios_out"
    result = extract_titles_from_files(base_path)

    # Print output clearly
    for level, titles in result.items():
        print(f"\n{level.upper()} ({len(titles)} titles):")
        for title in titles:
            print(f" - {title}")


if __name__ == "__main__":
    main()
