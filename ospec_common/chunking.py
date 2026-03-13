import json
import os

from fastchat.llm_judge.common import load_questions


def save_chunks_to_jsonl(question_file, chunk_size=500, output_dir="data_chunks", question_begin=None, question_end=None):
    questions = load_questions(question_file, question_begin, question_end)
    print(f"Loaded {len(questions)} questions, splitting into chunks of size {chunk_size}")

    os.makedirs(output_dir, exist_ok=True)

    chunk_files = []

    for i in range(0, len(questions), chunk_size):
        chunk = questions[i:i + chunk_size]
        chunk_idx = i // chunk_size

        chunk_file = os.path.join(output_dir, f"chunk_{chunk_idx:03d}.jsonl")

        with open(chunk_file, "w", encoding="utf-8") as file:
            for question in chunk:
                json.dump(question, file, ensure_ascii=False)
                file.write("\n")

        chunk_files.append(chunk_file)
        print(f"Saved chunk {chunk_idx}: {len(chunk)} questions to {chunk_file}")

    return chunk_files
