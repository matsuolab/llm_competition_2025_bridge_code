import sys
import json
import csv
from pathlib import Path
from collections import defaultdict

path_root = sys.argv[1]

HLE_GENRE_COUNT = {
    "Math": 1021,
    "Biology/Medicine": 280,
    "Computer Science/AI": 241,
    "Other": 233,
    "Physics": 230,
    "Humanities/Social Science": 219,
    "Chemistry": 165,
    "Engineering": 111,
}

all_hle_question = sum(list(HLE_GENRE_COUNT.values()))

result_files = Path(path_root).glob("**/results.jsonl")

summaries = []
counters = []
for rf in result_files:
    genre_counter = defaultdict(int)
    with rf.open() as f:
        for line in f:
            jline = json.loads(line)
            genre_counter[jline["category"]] += 1

    sf = rf.parent / "summary.json"
    with sf.open() as f:
        summary = json.load(f)
        print(summary["model_name"])
        print(sorted(genre_counter.items()))
        genre_counter["model_name"] = summary["model_name"]
        new_summary = {"model_name": summary["model_name"],
                "overall_accuracy": summary["overall_accuracy"] * summary["num_questions"] / all_hle_question}
        accuracy_per_category = summary["accuracy_per_category"]
        for k in HLE_GENRE_COUNT:
            accuracy_per_category[k] *= genre_counter[k] / HLE_GENRE_COUNT[k]
        new_summary.update(accuracy_per_category)
        new_summary["num_questions"] = summary["num_questions"]
        new_summary["timestamp"] = summary["timestamp"]
    summaries.append(new_summary)
    counters.append(genre_counter)

summary_keys = list(summaries[0].keys())
counter_keys = ["model_name"] + list(accuracy_per_category.keys())

with open("all_summaries_mod.csv", "w") as fO:
    writer = csv.DictWriter(fO, summary_keys)
    writer.writeheader()
    for s in summaries:
        writer.writerow(s)

with open("all_solved_count.csv", "w") as fO:
    writer = csv.DictWriter(fO, counter_keys)
    writer.writeheader()
    for c in counters:
        writer.writerow(c)

