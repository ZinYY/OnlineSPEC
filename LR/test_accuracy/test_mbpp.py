"""
Usage:
python test_mbpp.py --folder <folder path> --dataset <dataset path>
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import traceback
import signal
from contextlib import contextmanager


class TimeoutException(Exception):
    """Timeout exception"""
    pass


@contextmanager
def time_limit(seconds):
    """Limit code execution time"""
    def signal_handler(signum, frame):
        raise TimeoutException("Code execution timeout")
    
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def extract_python_code(text: str) -> Optional[str]:
    if not text:
        return None
    
    pattern1 = r'```python\s*\n(.*?)\n```'
    matches = re.findall(pattern1, text, re.DOTALL)
    if matches:
        return matches[0].strip()
    
    pattern2 = r'```\s*\n(.*?)\n```'
    matches = re.findall(pattern2, text, re.DOTALL)
    if matches:
        return matches[0].strip()
    
    lines = text.split('\n')
    code_start = -1
    for i, line in enumerate(lines):
        if line.strip().startswith('def ') or line.strip().startswith('import '):
            code_start = i
            break
    
    if code_start >= 0:
        return '\n'.join(lines[code_start:]).strip()
    
    return text.strip()


def run_test_cases(code: str, test_cases: list, test_setup_code: Optional[str] = None, reference_code: Optional[str] = None, timeout: int = 1) -> Dict[str, Any]:
    """
    Run test cases
    
    Args:
        code: Generated Python code
        test_cases: Test cases list
        test_setup_code: Test setup code (e.g. import statements, helper functions)
        reference_code: Reference answer code (for replacing function names)
        timeout: Timeout time (seconds)
    
    Returns:
        Result dictionary: {
            'passed': bool,
            'num_passed': int,
            'num_total': int,
            'error': str or None
        }
    """
    if not code:
        return {
            'passed': False,
            'num_passed': 0,
            'num_total': len(test_cases),
            'error': 'Code not found'
        }
    
    common_imports = """
import math
import re
import sys
import copy
import datetime
import itertools
import collections
import heapq
import statistics
import functools
import operator
import string
import random
from collections import Counter, defaultdict, OrderedDict, deque, namedtuple
from itertools import (combinations, permutations, product, chain, cycle, 
                       accumulate, groupby, islice, zip_longest, count, repeat)
from functools import reduce, lru_cache, partial
from operator import itemgetter, attrgetter, methodcaller
from math import sqrt, ceil, floor, gcd, factorial, log, log2, log10, exp
from heapq import heappush, heappop, heapify, nlargest, nsmallest, merge
from bisect import bisect_left, bisect_right, insort
from typing import List, Dict, Set, Tuple, Optional, Any
from sys import maxsize

class Node:
    '''Generic binary tree/linked list node class'''
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None
        self.next = None

class TreeNode:
    '''Binary tree node class (alias)'''
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class ListNode:
    '''Linked list node class'''
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
"""
    
    num_passed = 0
    num_total = len(test_cases)
    error_msg = None
    
    exec_globals = {}
    
    try:
        exec(common_imports, exec_globals)
        
        try:
            import regex
            exec_globals['regex'] = regex
        except ImportError:
            pass
    except Exception as e:
        pass
    
    try:
        if test_setup_code:
            with time_limit(timeout):
                exec(test_setup_code, exec_globals)
        
        with time_limit(timeout):
            exec(code, exec_globals)
    except TimeoutException:
        return {
            'passed': False,
            'num_passed': 0,
            'num_total': num_total,
            'error': f'Code execution timeout (may contain module-level infinite loops or time-consuming operations)'
        }
    except Exception as e:
        return {
            'passed': False,
            'num_passed': 0,
            'num_total': num_total,
            'error': f'Code execution failed: {type(e).__name__}: {str(e)}'
        }
    
    for i, test_case in enumerate(test_cases):
        try:
            with time_limit(timeout):
                exec(test_case, exec_globals)
                num_passed += 1
        except TimeoutException:
            error_msg = f'Test case {i+1} timeout'
            break
        except AssertionError as e:
            error_msg = f'Test case {i+1} failed: Assertion error'
            break
        except Exception as e:
            error_msg = f'Test case {i+1} failed: {type(e).__name__}: {str(e)}'
            break
    
    return {
        'passed': num_passed == num_total,
        'num_passed': num_passed,
        'num_total': num_total,
        'error': error_msg
    }


def load_mbpp_dataset(dataset_path: str) -> Dict[int, Dict]:
    dataset = {}
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                question_id = item.get('id') or item.get('task_id')
                dataset[question_id] = item
    
    return dataset


def evaluate_mbpp(results_folder: str, dataset_path: str, verbose: bool = False):
    print(f"\nLoading MBPP dataset: {dataset_path}")
    try:
        dataset = load_mbpp_dataset(dataset_path)
        print(f"Loaded {len(dataset)} questions")
    except Exception as e:
        print(f"Error: Failed to load dataset - {e}")
        return
    
    folder = Path(results_folder)
    json_files = []
    
    for file in folder.rglob("*.json"):
        if file.is_file():
            try:
                num = int(file.stem)
                rel_path = file.relative_to(folder)
                json_files.append((num, rel_path, file))
            except ValueError:
                continue
    
    json_files.sort(key=lambda x: (str(x[1].parent), x[0]))
    
    if not json_files:
        print(f"Warning: No JSON files found in folder {results_folder}")
        return
    
    print(f"\n{'='*80}")
    print(f"Evaluating MBPP")
    print(f"Results folder: {results_folder}")
    print(f"Found {len(json_files)} result files")
    print(f"{'='*80}\n")
    
    total = 0
    passed = 0
    errors = []
    
    for num, rel_path, json_file in json_files:
        total += 1
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                result = json.load(f)
            
            question_id = result.get('question_id') or result.get('task_id') or num
            
            if question_id not in dataset:
                print(f"[!] Warning: Question {question_id} not in dataset")
                errors.append((question_id, rel_path, "Question not in dataset"))
                continue
            
            dataset_item = dataset[question_id]
            test_cases = dataset_item.get('test_list', [])
            test_setup_code = dataset_item.get('test_setup_code', '')
            reference_code = dataset_item.get('answer', '') or dataset_item.get('code', '')
            
            if not test_cases:
                print(f"[!] Warning: Question {question_id} has no test cases")
                errors.append((question_id, rel_path, "No test cases"))
                continue
            
            generation_text = result.get('generation_text', '') or result.get('output', '')
            generated_code = extract_python_code(generation_text)
            
            test_result = run_test_cases(generated_code, test_cases, test_setup_code, reference_code)
            
            if test_result['passed']:
                passed += 1
                status = "✓"
            else:
                status = "✗"
                errors.append((question_id, rel_path, test_result['error']))
            
            if verbose or not test_result['passed']:
                print(f"[{status}] Question {question_id} ({rel_path})")
                print(f"    Passed: {test_result['num_passed']}/{test_result['num_total']}")
                if test_result['error']:
                    print(f"    Error: {test_result['error']}")
                if verbose and generated_code:
                    print(f"    Generated code:")
                    code_lines = generated_code.split('\n')
                    for line in code_lines[:10]:
                        print(f"      {line}")
                    if len(code_lines) > 10:
                        print(f"      ... (total {len(code_lines)} lines)")
                print()
        
        except Exception as e:
            print(f"[✗] Error processing file {rel_path}: {e}")
            traceback.print_exc()
            errors.append((num, rel_path, str(e)))
            continue
    
    accuracy = (passed / total * 100) if total > 0 else 0.0
    
    print(f"\n{'='*80}")
    print(f"MBPP evaluation results:")
    print(f"{'='*80}")
    print(f"Total questions: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Pass@1 accuracy: {accuracy:.2f}%")
    print(f"{'='*80}\n")
    
    if errors and not verbose:
        print(f"\nFailed questions ({len(errors)} questions):")
        print(f"{'-'*80}")
        for qid, path, error in errors[:20]:
            print(f"Question {qid} ({path}): {error}")
        if len(errors) > 20:
            print(f"... {len(errors) - 20} more errors")


def test_gold_answers(dataset_path: str, verbose: bool = False):
    print(f"\nLoading MBPP dataset: {dataset_path}")
    try:
        dataset = load_mbpp_dataset(dataset_path)
        print(f"Loaded {len(dataset)} questions")
    except Exception as e:
        print(f"Error: Failed to load dataset - {e}")
        return
    
    print(f"\n{'='*80}")
    print(f"Testing Gold answers in the dataset")
    print(f"Dataset: {dataset_path}")
    print(f"{'='*80}\n")
    
    total = 0
    passed = 0
    errors = []
    
    for question_id, item in sorted(dataset.items()):
        total += 1
        
        try:
            gold_code = item.get('answer') or item.get('code') or item.get('gold')
            test_cases = item.get('test_list', [])
            test_setup_code = item.get('test_setup_code', '')
            
            if not gold_code:
                print(f"[!] Question {question_id}: No gold answer found")
                errors.append((question_id, "No gold answer"))
                continue
            
            if not test_cases:
                print(f"[!] Question {question_id}: No test cases")
                errors.append((question_id, "No test cases"))
                continue
            
            test_result = run_test_cases(gold_code, test_cases, test_setup_code)
            
            if test_result['passed']:
                passed += 1
                status = "✓"
                if verbose:
                    print(f"[{status}] Question {question_id}: Passed {test_result['num_passed']}/{test_result['num_total']}")
            else:
                status = "✗"
                errors.append((question_id, test_result['error']))
                print(f"[{status}] Question {question_id}: Failed")
                print(f"    Passed: {test_result['num_passed']}/{test_result['num_total']}")
                print(f"    Error: {test_result['error']}")
                if verbose:
                    print(f"    Reference code:")
                    gold_lines = gold_code.split('\n')
                    for line in gold_lines[:15]:
                        print(f"      {line}")
                    if len(gold_lines) > 15:
                        print(f"      ... (total {len(gold_lines)} lines)")
                print()
        
        except Exception as e:
            print(f"[✗] Error processing question {question_id}: {e}")
            if verbose:
                traceback.print_exc()
            errors.append((question_id, str(e)))
            continue
    
    accuracy = (passed / total * 100) if total > 0 else 0.0
    
    print(f"\n{'='*80}")
    print(f"Gold answer test results:")
    print(f"{'='*80}")
    print(f"Total questions: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"{'='*80}\n")
    
    if errors:
        print(f"\nFailed questions ({len(errors)} questions):")
        print(f"{'-'*80}")
        for qid, error in errors[:30]:
            print(f"Question {qid}: {error}")
        if len(errors) > 30:
            print(f"... {len(errors) - 30} more errors")


def main():
    parser = argparse.ArgumentParser(description="MBPP code generation evaluation script")
    parser.add_argument(
        "--folder",
        type=str,
        help="Folder path containing result JSON files"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="MBPP dataset file path (JSONL format)"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed information"
    )
    parser.add_argument(
        "--test-gold",
        action="store_true",
        help="Test if the gold answers in the dataset can pass all test cases"
    )
    
    args = parser.parse_args()
    
    if args.test_gold:
        test_gold_answers(args.dataset, args.verbose)
    else:
        if not args.folder:
            parser.error("--folder is required (unless using --test-gold)")
        evaluate_mbpp(args.folder, args.dataset, args.verbose)


if __name__ == "__main__":
    main()
