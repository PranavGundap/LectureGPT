import os
import json

JSON_FOLDER = "data/jsons"

for file in os.listdir(JSON_FOLDER):

    if not file.endswith(".json"):
        continue

    path = os.path.join(JSON_FOLDER, file)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(f"Fixed: {file}")

print("✅ All JSON files converted to readable Unicode.")