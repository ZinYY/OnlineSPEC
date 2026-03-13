from datasets import load_dataset

input_path = "data/spider/raw/train.json"
output_dir = "data/spider/raw"
base_dataset = load_dataset("json", data_files=input_path)["train"]

split_data = base_dataset.train_test_split(test_size=.05, seed=42)
split_data["train"].to_json(f"{output_dir}/train.json")
split_data["test"].to_json(f"{output_dir}/val.json")