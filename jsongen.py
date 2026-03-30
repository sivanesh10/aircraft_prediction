import json

records = []

for i in range(1, 41):
    records.append({
        "engine_id": 1,
        "cycle": i,
        "setting_1": 0.5,
        "setting_2": 0.6,
        "setting_3": 0.7,
        "sensor_1": 518.7 + i*0.1,
        "sensor_2": 641.8 + i*0.1,
        "sensor_3": 1589.7 + i*0.1,
        "sensor_4": 1400.6 + i*0.1,
        "sensor_5": 14.6 + i*0.01,
        "sensor_6": 21.6 + i*0.01,
        "sensor_7": 554.3 + i*0.1,
        "sensor_8": 2388.0,
        "sensor_9": 9046.2,
        "sensor_10": 1.3,
        "sensor_11": 47.4,
        "sensor_12": 521.6,
        "sensor_13": 2388.0,
        "sensor_14": 8138.6,
        "sensor_15": 8.41,
        "sensor_16": 0.03,
        "sensor_17": 392,
        "sensor_18": 2388,
        "sensor_19": 100,
        "sensor_20": 39.0,
        "sensor_21": 23.4
    })

data = {"records": records}

# Convert to JSON file
with open("telemetry_test.json", "w") as f:
    json.dump(data, f, indent=2)

print("JSON file created: telemetry_test.json")