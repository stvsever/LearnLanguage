import json
from pathlib import Path
from typing import Any, Dict

def describe_metadata(data: Dict[str, Any]) -> Dict[str, Any]:
    topics = data.get("topics", [])
    num_topics = len(topics)

    # Use the primary (English) topic name as the key so it's JSON-serializable
    entries_per_topic = {
        topic["name"][0]: len(topic.get("entries", []))
        for topic in topics
    }

    total_entries = sum(entries_per_topic.values())
    average_entries = total_entries / num_topics if num_topics else 0

    # Find topic with most/fewest entries
    if entries_per_topic:
        max_topic = max(entries_per_topic, key=entries_per_topic.get)
        min_topic = min(entries_per_topic, key=entries_per_topic.get)
    else:
        max_topic = min_topic = None

    return {
        "num_topics": num_topics,
        "topic_names": list(entries_per_topic.keys()),
        "entries_per_topic": entries_per_topic,
        "total_entries": total_entries,
        "average_entries": average_entries,
        "topic_with_most_entries": {
            "name": max_topic,
            "count": entries_per_topic.get(max_topic, 0)
        },
        "topic_with_fewest_entries": {
            "name": min_topic,
            "count": entries_per_topic.get(min_topic, 0)
        }
    }

if __name__ == "__main__":
    path = Path(__file__).resolve().parent / "data" / "vocabulary_es.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    meta = describe_metadata(data)
    print(json.dumps(meta, indent=2, ensure_ascii=False))
