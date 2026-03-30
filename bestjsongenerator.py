import json

records = []

for i in range(1, 41):
    record = {
        "engine_id": 1,
        "cycle": i,
        "setting_1": 0.1 + i * 0.01,
        "setting_2": 0.2 + i * 0.01,
        "setting_3": 0.3 + i * 0.01,
    }

    for j in range(1, 22):
        base = 100 * j
        value = base + i * (j * 2)

        # 🔥 Add anomaly after cycle 30
        if i > 30:
            value *= 1.2

        record[f"sensor_{j}"] = round(value, 2)

    records.append(record)

data = {"records": records}

with open("test_input_best.json", "w") as f:
    json.dump(data, f, indent=2)

print("✅ JSON generated")