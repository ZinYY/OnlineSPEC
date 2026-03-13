from collections import Counter
from datasets import load_dataset
import os
import json
import random
import shutil
from pathlib import Path

LIMIT = 4800


def clear_dataset_cache(dataset_name):
    """Clear cache for a specific dataset to avoid version compatibility issues"""
    cache_dir = Path.home() / ".cache" / "huggingface" / "datasets"
    if cache_dir.exists():
        dataset_cache_name = dataset_name.replace("/", "___")
        for cache_path in cache_dir.glob(f"*{dataset_cache_name}*"):
            if cache_path.is_dir():
                print(f"Clearing cache: {cache_path}")
                try:
                    shutil.rmtree(cache_path)
                except Exception as error:
                    print(f"Warning: Could not remove cache {cache_path}: {error}")


class Collector:
    def __init__(self, name, *args, clear_cache=False, **kwargs) -> None:
        if clear_cache:
            clear_dataset_cache(name)

        max_retries = 2
        for attempt in range(max_retries):
            try:
                self.dataset = load_dataset(name, *args, **kwargs)
                break
            except (TypeError, ValueError, RuntimeError) as error:
                error_str = str(error)
                if ("must be called with a dataclass" in error_str or
                    "dataclass" in error_str.lower() or
                    "DatasetInfo" in error_str):
                    if attempt < max_retries - 1:
                        print(f"Cache compatibility issue detected for {name}. Clearing cache and retrying...")
                        clear_dataset_cache(name)
                        cache_dir = Path.home() / ".cache" / "huggingface" / "datasets"
                        if cache_dir.exists():
                            for cache_path in cache_dir.glob("*spider*"):
                                if cache_path.is_dir():
                                    try:
                                        shutil.rmtree(cache_path)
                                        print(f"Cleared additional cache: {cache_path}")
                                    except Exception:
                                        pass
                    else:
                        raise RuntimeError(f"Failed to load dataset {name} after clearing cache. "
                                           f"Original error: {error}\n"
                                           f"Try manually clearing cache: rm -rf ~/.cache/huggingface/datasets/*spider*")
                else:
                    raise
        self.name = name.replace("/", "_")

    def get_raw_filename(self, split, prefix):
        if prefix is not None:
            return f"raw_data/{prefix}_{self.name}_{split}_raw.json"
        return f"raw_data/{self.name}_{split}_raw.json"

    def get_output_filename(self, split, prefix):
        if prefix is not None:
            return f"raw_data/{prefix}_{self.name}_{split}.json"
        return f"raw_data/{self.name}_{split}.json"

    def collect(self,
                splits,
                transform,
                size=None, prefix=None,
                split_train=False):
        def load_case(filename):
            with open(rawfile, "r") as file:
                raw_data = list(file)

                cases = []
                index = 0
                for case in raw_data:
                    case = json.loads(case)
                    cases.append(case)
                    index += 1
                    if index == LIMIT:
                        break
            return cases

        splits = [splits] if isinstance(splits, str) else splits
        cases = []
        for split in splits:
            rawfile = self.get_raw_filename(split, prefix)
            self.dataset[split].to_json(rawfile)
            cases += load_case(rawfile)
            os.remove(rawfile)

        split = "train" if len(splits) > 1 else splits[0]
        cases = [transform(i, case) for i, case in enumerate(cases)]
        cases = [c for c in cases if (c is not None) and ("conversation" in c)]
        cases = [{k: c[k] for k in ["id", "conversation"]} for c in cases]
        if size is not None:
            random.shuffle(cases)
            cases = cases[:size]

        if split_train:
            assert split == "train"
            split_eval_cases = cases[:200]
            split_train_cases = cases[200:]
            with open(self.get_output_filename("train", prefix), "w") as file:
                print(f"Train length: {len(split_eval_cases)}")
                json.dump(split_train_cases, file)
            with open(self.get_output_filename("eval", prefix), "w") as file:
                print(f"Eval length: {len(split_eval_cases)}")
                json.dump(split_eval_cases, file)
        else:
            with open(self.get_output_filename(split, prefix), "w") as file:
                json.dump(cases, file)

        print(f"Dataset: {self.name}, Split: {split}, Length: {len(cases)}")
        raw_texts = [c["conversation"][0]["content"] for c in cases]
        self.count_unique_tokens(raw_texts)

    def count_unique_tokens(self, dataset):
        token_counter = Counter()
        for text in dataset:
            tokens = text.split(" ")
            token_counter.update(tokens)
        unique_tokens = set(token_counter.keys())
        num_unique_tokens = len(unique_tokens)
        print(f"Number of unique tokens {num_unique_tokens}")
        return num_unique_tokens


def transform(i, case, need_label=False):
    sql_prompt = "Could you translate the following question into SQL. Please only generate SQL, don't include explanation in the answer. "
    case["id"] = f"identity_{i}"
    if need_label:
        case["conversation"] = [
            {
                "role": "user",
                "content": sql_prompt + case["question"],
            },
            {
                "role": "assistant",
                "content": " ".join(case["query_toks_no_value"]),
            },
        ]
    else:
        case["conversation"] = [
            {
                "role": "user",
                "content": sql_prompt + case["question"],
            }
        ]
    return case


def transform_gsm8k(i, case):
    return {
        "id": f"identity_{i}",
        "conversations": [
            {
                "from": "human",
                "value": case["question"],
            },
            {
                "from": "gpt",
                "value": case["answer"],
            },
        ],
    }


def transform_spider(i, case):
    sql_prompt = "Could you translate the following question into SQL. Please only generate SQL, don't include explanation in the answer. "

    query = case.get("query_toks_no_value", "")
    if isinstance(query, list):
        query = " ".join(query)

    return {
        "id": f"identity_{i}",
        "conversations": [
            {
                "from": "human",
                "value": sql_prompt + case.get("question", ""),
            },
            {
                "from": "gpt",
                "value": query,
            },
        ],
    }


def transform_code_search(i, case):
    code_prompt = "Please generate code based on the following doc:\n"

    return {
        "id": f"identity_{i}",
        "conversations": [
            {
                "from": "human",
                "value": code_prompt + case.get("func_documentation_string", ""),
            },
            {
                "from": "gpt",
                "value": case.get("func_code_string", ""),
            },
        ],
    }


def transform_finance(i, case):
    return {
        "id": f"identity_{i}",
        "conversations": [
            {
                "from": "human",
                "value": case.get("instruction", ""),
            },
            {
                "from": "gpt",
                "value": case.get("output", ""),
            },
        ],
    }


def save_jsonl(data, output_file, max_items=None, start_idx=0):
    if start_idx > 0:
        data = data[start_idx:]

    if max_items is not None:
        data = data[:max_items]

    with open(output_file, "w", encoding="utf-8") as file:
        for item in data:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Saved {len(data)} items to {output_file} (from index {start_idx})")


def main():
    print("=" * 60)
    print("Processing GSM8K dataset...")
    print("=" * 60)
    c_gsm8k = Collector("gsm8k", "main")
    dataset_gsm8k = c_gsm8k.dataset

    all_gsm8k_cases = []

    if "train" in dataset_gsm8k:
        print(f"Processing train split, {len(dataset_gsm8k['train'])} items...")
        for i, case in enumerate(dataset_gsm8k["train"]):
            transformed = transform_gsm8k(i, case)
            all_gsm8k_cases.append(transformed)

    if "test" in dataset_gsm8k:
        print(f"Processing test split, {len(dataset_gsm8k['test'])} items...")
        start_idx = len(all_gsm8k_cases)
        for i, case in enumerate(dataset_gsm8k["test"]):
            transformed = transform_gsm8k(start_idx + i, case)
            all_gsm8k_cases.append(transformed)

    save_jsonl(all_gsm8k_cases, "gsm8k_offline.jsonl", max_items=800, start_idx=0)
    save_jsonl(all_gsm8k_cases, "gsm_4k.jsonl", max_items=4000, start_idx=800)

    print("\n" + "=" * 60)
    print("Processing Spider dataset...")
    print("=" * 60)
    c_spider = Collector("spider")
    dataset_spider = c_spider.dataset

    all_spider_cases = []

    for split_name in dataset_spider.keys():
        print(f"Processing {split_name} split, {len(dataset_spider[split_name])} items...")
        start_idx = len(all_spider_cases)
        for i, case in enumerate(dataset_spider[split_name]):
            transformed = transform_spider(start_idx + i, case)
            all_spider_cases.append(transformed)

    save_jsonl(all_spider_cases, "spider_offline.jsonl", max_items=800, start_idx=0)
    save_jsonl(all_spider_cases, "spider_online_4k.jsonl", max_items=4000, start_idx=800)

    print("\n" + "=" * 60)
    print("Processing Code Search dataset...")
    print("=" * 60)

    try:
        c_code_search = Collector("code-search-net/code_search_net", "python")
        dataset_code_search = c_code_search.dataset
    except RuntimeError as error:
        if "Dataset scripts are no longer supported" in str(error):
            print("ERROR: The code_search_net dataset uses old dataset scripts format.")
            print("Solution: Please downgrade datasets library to version 2.x:")
            print("  pip install 'datasets<3.0.0'")
            print("\nAlternatively, you can skip this dataset or use an alternative.")
            print("Skipping code_search dataset processing...")
            dataset_code_search = None
        else:
            raise

    if dataset_code_search is not None:
        all_code_search_cases = []

        if "train" in dataset_code_search:
            print(f"Processing train split, {len(dataset_code_search['train'])} items...")
            for i, case in enumerate(dataset_code_search["train"]):
                transformed = transform_code_search(i, case)
                all_code_search_cases.append(transformed)

        if "test" in dataset_code_search:
            print(f"Processing test split, {len(dataset_code_search['test'])} items...")
            start_idx = len(all_code_search_cases)
            for i, case in enumerate(dataset_code_search["test"]):
                transformed = transform_code_search(start_idx + i, case)
                all_code_search_cases.append(transformed)

        save_jsonl(all_code_search_cases, "code_search_offline.jsonl", max_items=800, start_idx=0)
        save_jsonl(all_code_search_cases, "code_search_online_4k.jsonl", max_items=4000, start_idx=800)

    print("\n" + "=" * 60)
    print("Processing Finance dataset...")
    print("=" * 60)
    c_finance = Collector("gbharti/finance-alpaca")
    dataset_finance = c_finance.dataset

    all_finance_cases = []

    if "train" in dataset_finance:
        print(f"Processing train split, {len(dataset_finance['train'])} items...")
        for i, case in enumerate(dataset_finance["train"]):
            transformed = transform_finance(i, case)
            all_finance_cases.append(transformed)

    for split_name in dataset_finance.keys():
        if split_name != "train":
            print(f"Processing {split_name} split, {len(dataset_finance[split_name])} items...")
            start_idx = len(all_finance_cases)
            for i, case in enumerate(dataset_finance[split_name]):
                transformed = transform_finance(start_idx + i, case)
                all_finance_cases.append(transformed)

    save_jsonl(all_finance_cases, "finance_offline.jsonl", max_items=800, start_idx=0)
    save_jsonl(all_finance_cases, "finance_online_4k.jsonl", max_items=4000, start_idx=800)

    print("\n" + "=" * 60)
    print("All datasets processed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
