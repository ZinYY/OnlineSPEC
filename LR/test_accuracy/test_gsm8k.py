"""
Usage:
python test_gsm8k.py --folder <folder path> --use-llm --llm-port 6767
"""

import argparse
import json
import re
from pathlib import Path
import asyncio
from openai import AsyncOpenAI


def extract_boxed_content(text):
    if not text:
        return None
    
    pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    
    matches = re.findall(pattern, text)
    
    if matches:
        return matches[-1].strip()
    
    return None


def normalize_answer(answer):
    if answer is None:
        return None
    
    answer = answer.replace(' ', '')
    
    answer = answer.replace('dfrac', 'frac')
    
    answer = answer.replace('\\,', '')
    answer = answer.replace('\\!', '')
    answer = answer.replace('\\:', '')
    answer = answer.replace('\\;', '')
    
    answer = answer.replace('\\left', '')
    answer = answer.replace('\\right', '')
    
    return answer.lower()


async def get_model_response(prompt, model="Qwen/Qwen2.5-7B-Instruct", temperature=0.0, max_tokens=100, stop=None, client=None):
    try:
        response = await client.completions.create(
            model=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            top_p=0.95
        )
        return response.choices[0].text.strip()
    except Exception as e:
        print(f"Error getting response: {e} {model}")
        return None


async def compare_with_llm(model_answer, gold_answer, client):
    prompt = f"""Compare these two answers and determine if they contain the same numerical values or mathematical content.

IGNORE:
- All formatting (spaces, commas, LaTeX commands like \\text, \\left, \\right, etc.)
- Units (dollars, meters, etc.)
- Currency symbols ($, €, etc.)
- Connecting words ("and", commas between numbers)
- Variable names (x =, y =, etc.)

FOCUS ONLY ON:
- The actual numbers or mathematical expressions

Examples:
- "10" and "$ 10" → YES (same number)
- "19, 43" and "19 \\text{{ and }}43" → YES (same numbers)
- "312" and "312 \\text{{ dollars}}" → YES (same number)
- "(3/5, 8/3]" and "(3/5,8/3]" → YES (same interval, just spacing difference)

Answer 1: {model_answer}
Answer 2: {gold_answer}

Do these contain the same numerical/mathematical content? Answer ONLY "YES" or "NO":"""
    
    response = await get_model_response(prompt, client=client, max_tokens=10)
    
    if response is None:
        return False
    
    response_lower = response.lower().strip()
    return "yes" in response_lower


async def compare_answers(model_answer, gold_answer, client=None):
    if model_answer is None:
        return False, "no_answer"
    
    model_normalized = normalize_answer(model_answer)
    gold_normalized = normalize_answer(gold_answer)
    
    if gold_normalized is None:
        return False, "no_gold"
    
    if model_normalized == gold_normalized:
        return True, "exact_match"
    
    if client is not None:
        is_correct = await compare_with_llm(model_answer, gold_answer, client)
        return is_correct, "llm_judge"
    else:
        return False, "no_llm"


async def evaluate_folder(folder_path, verbose=False, client=None):
    folder = Path(folder_path)
    
    if not folder.exists():
        return 0, 0, 0.0
    
    json_files = []
    for file in folder.rglob("*.json"):
        if file.is_file():
            try:
                num = int(file.stem)
                rel_path = file.relative_to(folder)
                json_files.append((num, rel_path, file))
            except ValueError:
                rel_path = file.relative_to(folder)
                json_files.append((None, rel_path, file))
    
    json_files.sort(key=lambda x: (str(x[1].parent), x[0]))
    
    if not json_files:
        return 0, 0, 0.0
    
    total = 0
    correct = 0
    
    method_stats = {
        "exact_match": 0,
        "llm_judge": 0,
        "no_answer": 0,
        "no_gold": 0,
        "no_llm": 0
    }
    
    print(f"\n{'='*80}")
    print(f"Evaluating folder: {folder_path}")
    print(f"Found {len(json_files)} JSON files (recursive search)")
    if client:
        print(f"LLM judgment: Enabled (local vllm)")
    else:
        print(f"LLM judgment: Disabled")
    print(f"{'='*80}\n")
    
    current_subfolder = None
    
    for num, rel_path, json_file in json_files:
        total += 1
        
        subfolder = str(rel_path.parent)
        if subfolder != current_subfolder:
            current_subfolder = subfolder
            if current_subfolder != '.':
                print(f"\n--- {current_subfolder} ---")
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            generation_text = data['generation_text']
            gold_answer = data['gold']
            
            model_answer = extract_boxed_content(generation_text)
            
            gold_boxed = extract_boxed_content(gold_answer)
            if gold_boxed is None and gold_answer:
                gold_boxed = gold_answer.strip()
            
            is_correct, method = await compare_answers(model_answer, gold_boxed, client)
            method_stats[method] += 1
            
            if is_correct:
                correct += 1
                status = "✓"
            else:
                status = "✗"
            
            if verbose:
                question_id = data['question_id']
                print(f"[{status}] Question {question_id} ({rel_path}) [{method}]")
                print(f"  Model answer: {model_answer}")
                print(f"  Gold answer: {gold_boxed}")
                if not is_correct:
                    print(f"  Status: Error")
                print()
            elif not is_correct:
                question_id = data['question_id']
                print(f"[✗] Question {question_id} ({rel_path}) [{method}]: Model={model_answer}, Gold={gold_boxed}")
        
        except Exception as e:
            print(f"Error processing file {rel_path}: {e}")
            continue
    
    accuracy = (correct / total * 100) if total > 0 else 0.0
    
    print(f"\n{'='*80}")
    print(f"Method statistics:")
    print(f"  Exact match (exact_match): {method_stats['exact_match']}")
    print(f"  LLM judgment (llm_judge): {method_stats['llm_judge']}")
    print(f"  No model answer (no_answer): {method_stats['no_answer']}")
    print(f"  No gold answer (no_gold): {method_stats['no_gold']}")
    print(f"  No LLM (no_llm): {method_stats['no_llm']}")
    print(f"{'='*80}")
    
    return total, correct, accuracy


async def evaluate_by_subfolder(folder_path, verbose=False, client=None):
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"Folder {folder_path} does not exist")
        return {}
    
    subfolder_files = {}
    
    for file in folder.rglob("*.json"):
        if file.is_file():
            try:
                num = int(file.stem)
                rel_path = file.relative_to(folder)
                subfolder = str(rel_path.parent)
                
                if subfolder not in subfolder_files:
                    subfolder_files[subfolder] = []
                
                subfolder_files[subfolder].append((num, rel_path, file))
            except ValueError:
                continue
    
    for subfolder in subfolder_files:
        subfolder_files[subfolder].sort(key=lambda x: x[0])
    
    results = {}
    
    print(f"\n{'='*80}")
    print(f"Evaluating by subfolder: {folder_path}")
    if client:
        print(f"LLM judgment: Enabled (local vllm)")
    else:
        print(f"LLM judgment: Disabled")
    print(f"{'='*80}\n")
    
    for subfolder, files in sorted(subfolder_files.items()):
        print(f"\n--- {subfolder if subfolder != '.' else 'Root'} ---")
        
        total = 0
        correct = 0
        
        for num, rel_path, json_file in files:
            total += 1
            
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                generation_text = data['generation_text']
                gold_answer = data['gold']
                
                model_answer = extract_boxed_content(generation_text)
                gold_boxed = extract_boxed_content(gold_answer)
                
                is_correct, method = await compare_answers(model_answer, gold_boxed, client)
                
                if is_correct:
                    correct += 1
                    status = "✓"
                else:
                    status = "✗"
                
                if verbose:
                    question_id = data['question_id']
                    print(f"[{status}] Question {question_id} ({json_file.name}) [{method}]")
                    print(f"  Model answer: {model_answer}")
                    print(f"  Gold answer: {gold_boxed}")
                    if not is_correct:
                        print(f"  Status: Error")
                    print()
                elif not is_correct:
                    question_id = data['question_id']
                    print(f"[✗] Question {question_id} ({rel_path}) [{method}]: Model={model_answer}, Gold={gold_boxed}")
            
            except Exception as e:
                print(f"Error processing file {rel_path}: {e}")
                continue
        
        accuracy = (correct / total * 100) if total > 0 else 0.0
        results[subfolder] = (total, correct, accuracy)
        
        print(f"  Total: {correct}/{total} = {accuracy:.2f}%")
    
    return results


async def main_async():
    parser = argparse.ArgumentParser(description="Calculate model answer accuracy")
    parser.add_argument(
        "--folder",
        type=str,
        required=True,
        help="Folder path containing JSON files (recursive search)"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed information for each question"
    )
    parser.add_argument(
        "--by-subfolder",
        "-s",
        action="store_true",
        help="Calculate accuracy by subfolder"
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use local vllm model for answer comparison"
    )
    parser.add_argument(
        "--llm-port",
        type=int,
        default=6767,
        help="Local vllm service port (default: 6767)"
    )
    
    args = parser.parse_args()
    
    client = None
    if args.use_llm:
        try:
            client = AsyncOpenAI(
                base_url=f"http://localhost:{args.llm_port}/v1",
                api_key="dummy"
            )
            print(f"Connected to local vllm service (port {args.llm_port})")
        except Exception as e:
            print(f"Cannot connect to local vllm service: {e}")
            print(f"Continue using only exact match mode...")
    
    if args.by_subfolder:
        results = await evaluate_by_subfolder(args.folder, verbose=args.verbose, client=client)
        
        print(f"\n{'='*80}")
        print(f"Summary results:")
        print(f"{'='*80}")
        
        total_all = 0
        correct_all = 0
        
        for subfolder, (total, correct, accuracy) in sorted(results.items()):
            display_name = subfolder if subfolder != '.' else 'Root'
            print(f"{display_name:30s}: {correct:4d}/{total:4d} = {accuracy:6.2f}%")
            total_all += total
            correct_all += correct
        
        print(f"{'-'*80}")
        overall_accuracy = (correct_all / total_all * 100) if total_all > 0 else 0.0
        print(f"{'Total':30s}: {correct_all:4d}/{total_all:4d} = {overall_accuracy:6.2f}%")
        print(f"{'='*80}\n")
    else:
        total, correct, accuracy = await evaluate_folder(args.folder, verbose=args.verbose, client=client)
        
        print(f"\n{'='*80}")
        print(f"Evaluation results:")
        print(f"{'='*80}")
        print(f"Total questions: {total}")
        print(f"Correct answers: {correct}")
        print(f"Incorrect answers: {total - correct}")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"{'='*80}\n")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
