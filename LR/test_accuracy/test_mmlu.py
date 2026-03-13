"""
Usage:
python test_mmlu.py --folder <folder path>
"""

import argparse
import json
import re
from pathlib import Path


def extract_boxed_content(text):
    if not text:
        return None
    
    pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    
    matches = re.findall(pattern, text)
    
    if matches:
        return matches[-1].strip()
    
    return None



def compare_answers(model_answer, gold_answer):
    """
    Compare model answer and gold answer
    """
    for letter in model_answer:
        if letter in ['A', 'B', 'C', 'D', 'a', 'b', 'c', 'd']:
            model_answer = letter.lower()
            break
    return model_answer == gold_answer.lower()


def evaluate_folder(folder_path, verbose=False):
    """
    Recursively evaluate the accuracy of all JSON files in the folder
    """
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"Error: Folder {folder_path} does not exist")
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
    
    total = 0
    correct = 0
    print(f"\n{'='*80}")
    print(f"Evaluating folder: {folder_path}")
    print(f"Found {len(json_files)} JSON files (including subfolders)")
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
            
            generation_text = data.get('generation_text', '')
            gold_answer = data.get('gold', '')
            
            model_answer = extract_boxed_content(generation_text)
            
            is_correct = compare_answers(model_answer, gold_answer)
            
            if is_correct:
                correct += 1
                status = "✓"
            else:
                status = "✗"
            
            if verbose:
                question_id = data.get('question_id', num)
                print(f"[{status}] Question {question_id} ({rel_path})")
                print(f"  Model answer: {model_answer}")
                print(f"  Gold answer: {gold_answer}")
                if not is_correct:
                    print(f"  Status: Error")
                print()
            elif not is_correct:
                question_id = data.get('question_id', num)
                print(f"[✗] Question {question_id} ({rel_path}): Model={model_answer}, Gold={gold_answer}")
        
        except Exception as e:
            print(f"Error processing file {rel_path}: {e}")
            continue
    
    accuracy = (correct / total * 100) if total > 0 else 0.0
    
    return total, correct, accuracy


def evaluate_by_subfolder(folder_path, verbose=False):
    """
    Evaluate the accuracy of all JSON files in the folder by subfolders
    """
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"Error: Folder {folder_path} does not exist")
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
    print(f"Evaluating by subfolders: {folder_path}")
    print(f"{'='*80}\n")
    
    for subfolder, files in sorted(subfolder_files.items()):
        print(f"\n--- Evaluating {subfolder if subfolder != '.' else 'Root'} ---")
        
        total = 0
        correct = 0
        
        for num, rel_path, json_file in files:
            total += 1
            
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                generation_text = data.get('generation_text', '')
                gold_answer = data.get('gold', '')
                
                model_answer = extract_boxed_content(generation_text)
                
                is_correct = compare_answers(model_answer, gold_answer)
                
                if is_correct:
                    correct += 1
                    status = "✓"
                else:
                    status = "✗"
                
                if verbose:
                    question_id = data.get('question_id', num)
                    print(f"[{status}] Question {question_id} ({json_file.name})")
                    print(f"  Model answer: {model_answer}")
                    print(f"  Gold answer: {gold_answer}")
                    if not is_correct:
                        print(f"  Status: Error")
                    print()
                elif not is_correct:
                    question_id = data.get('question_id', num)
                    print(f"[✗] Question {question_id} ({rel_path}): Model={model_answer}, Gold={gold_answer}")
            
            except Exception as e:
                print(f"Error processing file {rel_path}: {e}")
                continue
        
        accuracy = (correct / total * 100) if total > 0 else 0.0
        results[subfolder] = (total, correct, accuracy)
        
        print(f"  Subtotal: {correct}/{total} = {accuracy:.2f}%")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Calculate the accuracy of model answers")
    parser.add_argument(
        "--folder",
        type=str,
        required=True,
        help="Folder path containing JSON files (supports recursive search)"
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
        help="Evaluate by subfolders separately"
    )
    
    args = parser.parse_args()
    
    if args.by_subfolder:
        results = evaluate_by_subfolder(args.folder, verbose=args.verbose)
        
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
        total, correct, accuracy = evaluate_folder(args.folder, verbose=args.verbose)
        
        print(f"\n{'='*80}")
        print(f"Evaluation results:")
        print(f"{'='*80}")
        print(f"Total questions: {total}")
        print(f"Correct answers: {correct}")
        print(f"Incorrect answers: {total - correct}")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
