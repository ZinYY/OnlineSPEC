import json
import re
from datasets import load_dataset


def extract_function_signature(code: str) -> str:
    if not code:
        return ""
    
    pattern = r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\):'
    match = re.search(pattern, code)
    
    if match:
        return match.group(0)
    
    return ""


def process_mbpp(output_path):
    print("Downloading MBPP dataset from Hugging Face...")
    dataset = load_dataset("google-research-datasets/mbpp", "full")

    print(f"Dataset contains splits: {list(dataset.keys())}")
    
    processed_data = []
    
    splits_to_process = ['prompt', 'test', 'validation', 'train']
    
    skipped_count = 0
    
    for split_name in splits_to_process:
        if split_name in dataset:
            print(f"Processing {split_name} split, {len(dataset[split_name])} items...")
            for entry in dataset[split_name]:
                code = entry["code"]
                function_signature = extract_function_signature(code)
                
                if not function_signature:
                    skipped_count += 1
                    print(f"  Skipped: task_id={entry['task_id']} (cannot extract function signature)")
                    continue
                
                original_question = entry["text"]
                enhanced_question = f"{original_question}\n\nPlease implement a function with the following signature:\n{function_signature}"
                
                processed_data.append({
                    "id": entry["task_id"],
                    "question": enhanced_question,
                    "original_question": original_question,
                    "function_signature": function_signature,
                    "answer": code,
                    "test_list": entry.get("test_list", []),
                    "test_setup_code": entry.get("test_setup_code", ""),
                    "challenge_test_list": entry.get("challenge_test_list", [])
                })
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in processed_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    print(f"\nProcessing completed, file saved to: {output_path}")
    print(f"Saved: {len(processed_data)} items")
    if skipped_count > 0:
        print(f"Skipped: {skipped_count} items (cannot extract function signature)")
    print(f"Save rate: {len(processed_data)/(len(processed_data)+skipped_count)*100:.1f}%")
    
    if processed_data:
        print("\nExample data (first item):")
        example = processed_data[0].copy()
        if 'question' in example and len(example['question']) > 200:
            example['question'] = example['question'][:200] + "..."
        if 'answer' in example and len(example['answer']) > 200:
            example['answer'] = example['answer'][:200] + "..."
        print(json.dumps(example, indent=2, ensure_ascii=False))
        print(f"  - Test case number: {len(processed_data[0].get('test_list', []))}")
        print(f"  - Function signature: {processed_data[0].get('function_signature', 'N/A')}")
    test_counts = [len(item.get('test_list', [])) for item in processed_data]
    
    if test_counts:
        import statistics
        print(f"\nData quality statistics:")
        print(f"  - Average test case number: {statistics.mean(test_counts):.2f}")
        print(f"  - Minimum test case number: {min(test_counts)}")
        print(f"  - Maximum test case number: {max(test_counts)}")
        print(f"  - All data contains valid function signature ✓")

def process_math(output_path):
    print("Downloading MATH dataset from Hugging Face...")
    try:
        dataset = load_dataset("qwedsacf/competition_math", trust_remote_code=True)
        
        processed_data = []
        for split in dataset.keys():
            for i, entry in enumerate(dataset[split]):
                if entry["type"] == "Algebra":  # only process Algebra
                    processed_data.append({
                        "id": f"math_{split}_{i}",
                        "question": entry["problem"],
                        "answer": entry["solution"]
                    })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in processed_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
                
        print(f"Processing completed, file saved to: {output_path}")
        print(f"Saved: {len(processed_data)} items")
    except Exception as e:
        print(f"Error downloading or processing MATH dataset: {e}")

def process_gsm8k(output_path):
    print("Downloading GSM8K dataset from Hugging Face...")
    try:
        dataset = load_dataset("gsm8k", "main")
        
        print(f"Dataset contains splits: {list(dataset.keys())}")
        
        processed_data = []
        
        for split_name in ['train', 'test']:
            if split_name in dataset:
                print(f"Processing {split_name} split, {len(dataset[split_name])} items...")
                for idx, entry in enumerate(dataset[split_name]):
                    if len(processed_data) >= 2500:
                        break
                    
                    question = entry.get("question", "")
                    answer_full = entry.get("answer", "")
                    
                    answer = ""
                    if "####" in answer_full:
                        parts = answer_full.split("####")
                        if len(parts) > 1:
                            answer = parts[-1].strip()
                    
                    if not answer:
                        continue
                    
                    processed_data.append({
                        "id": len(processed_data) + 1,
                        "question": question,
                        "answer": answer
                    })
                
                if len(processed_data) >= 2500:
                    break
        
        processed_data = processed_data[:2500]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in processed_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"\nProcessing completed, file saved to: {output_path}")
        print(f"Saved: {len(processed_data)} items")
        
        if processed_data:
            print("\nExample data (first item):")
            example = processed_data[0].copy()
            if len(example.get('question', '')) > 200:
                example['question'] = example['question'][:200] + "..."
            if len(example.get('answer', '')) > 200:
                example['answer'] = example['answer'][:200] + "..."
            print(json.dumps(example, indent=2, ensure_ascii=False))
            
    except Exception as e:
        print(f"Error downloading or processing GSM2.5K dataset: {e}")
        import traceback
        traceback.print_exc()

def process_mmlu(output_path, subjects=None):
    print("Downloading MMLU dataset from Hugging Face...")
    
    try:
        dataset = load_dataset("cais/mmlu", "all")
        
        print(f"Dataset contains splits: {list(dataset.keys())}")
        
        processed_data = []
        subject_stats = {}
        
        for split_name in dataset.keys():
            print(f"\nProcessing {split_name} split, {len(dataset[split_name])} items...")
            
            for idx, entry in enumerate(dataset[split_name]):
                subject = entry.get("subject", "unknown")
                
                if subjects and subject not in subjects:
                    continue
                
                if subject not in subject_stats:
                    subject_stats[subject] = {"dev": 0, "validation": 0, "test": 0, "auxiliary_train": 0}
                subject_stats[subject][split_name] = subject_stats[subject].get(split_name, 0) + 1
                
                question = entry.get("question", "")
                choices = entry.get("choices", [])
                answer_idx = entry.get("answer", 0)
                
                formatted_question = f"**Question:** {question}\n\n**Choices:**\n"
                choice_letters = ['A', 'B', 'C', 'D']
                for i, choice in enumerate(choices):
                    formatted_question += f"{choice_letters[i]}. {choice}\n"
                
                formatted_question += f"\nOnly one answer is correct. Please select the correct answer letter and put it in \\boxed{{}}."
                
                answer_letter = choice_letters[answer_idx] if answer_idx < len(choice_letters) else "A"
                
                processed_data.append({
                    "id": f"{subject}_{split_name}_{idx}",
                    "subject": subject,
                    "question": formatted_question,
                    "answer": answer_letter
                })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in processed_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"\n{'='*60}")
        print(f"Processing completed, file saved to: {output_path}")
        print(f"Saved: {len(processed_data)} items")
        
        if subject_stats:
            print(f"\n{'='*60}")
            print(f"Subject statistics (Total {len(subject_stats)} subjects):")
            print(f"{'='*60}")
            
            for subject in sorted(subject_stats.keys()):
                stats = subject_stats[subject]
                total = sum(stats.values())
                print(f"\n{subject}:")
                print(f"  Total: {total} items")
                for split in ['auxiliary_train', 'dev', 'validation', 'test']:
                    if stats.get(split, 0) > 0:
                        print(f"  - {split}: {stats[split]} items")
            
            print(f"\n{'='*60}")
            print(f"Total statistics:")
            split_totals = {}
            for stats in subject_stats.values():
                for split, count in stats.items():
                    split_totals[split] = split_totals.get(split, 0) + count
            
            for split in ['auxiliary_train', 'dev', 'validation', 'test']:
                if split in split_totals:
                    print(f"  - {split}: {split_totals[split]} items")
            print(f"  - Total: {sum(split_totals.values())} items")
        
        if processed_data:
            print(f"\n{'='*60}")
            print("Example data (first item):")
            print(f"{'='*60}")
            example = processed_data[0].copy()
            example_display = {
                "id": example.get("id"),
                "subject": example.get("subject"),
                "split": example.get("split"),
                "question_text": example.get("question_text", "")[:150] + "..." if len(example.get("question_text", "")) > 150 else example.get("question_text", ""),
                "choices": example.get("choices", []),
                "answer_letter": example.get("answer_letter"),
                "correct_choice": example.get("correct_choice", "")
            }
            print(json.dumps(example_display, indent=2, ensure_ascii=False))
            
            print(f"\nFull formatted question preview:")
            print(example.get("question", "")[:400] + "..." if len(example.get("question", "")) > 400 else example.get("question", ""))
        
    except Exception as e:
        print(f"Error downloading or processing MMLU dataset: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    process_mmlu(
        output_path="mmlu.jsonl",
        subjects=["college_mathematics", "high_school_mathematics"]
    )
    process_mbpp(output_path="mbpp.jsonl")
    process_math(output_path="math.jsonl")
    process_gsm8k(output_path="gsm2.5k.jsonl")