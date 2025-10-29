"""
Benchmark script for DeepReviewer using WestlakeNLP/DeepReview-13K dataset
"""

import json
import os
import random
import argparse
from datasets import load_dataset
from ai_researcher import DeepReviewer
import time
from datetime import datetime
# Set CUDA device if needed
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

# Configuration
RANDOM_SEED = 42
NUM_SAMPLES = 100

# Set random seed for reproducibility
random.seed(RANDOM_SEED)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Benchmark DeepReviewer using WestlakeNLP/DeepReview-13K dataset"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="WestlakeNLP/DeepReviewer-7B",
        help="Model name to use (default: WestlakeNLP/DeepReviewer-7B)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="fast_mode",
        choices=["base_mode", "fast_mode", "standard_mode", "best_mode"],
        help="Evaluation mode to use (default: fast_mode)"
    )
    return parser.parse_args()


def main(args):

    # 1. Load WestlakeNLP/DeepReview-13K dataset (test split only)
    print("Loading DeepReview-13K dataset (test split)...")
    ds = load_dataset("WestlakeNLP/DeepReview-13K", split="test")

    # 2. Extract paper context from data and set as an extra column named as paper_text
    print("Extracting paper text from dataset...")

    def extract_paper_text(example):
        """Extract paper content from the user role in inputs"""
        # The inputs field contains a list of messages
        # Extract the content from the message with role "user"
        inputs = example.get('inputs', [])
        paper_text = ""

        # If inputs is a string, parse it as JSON
        if isinstance(inputs, str):
            try:
                inputs = json.loads(inputs)
            except json.JSONDecodeError:
                inputs = []

        if isinstance(inputs, list):
            for message in inputs:
                if isinstance(message, dict) and message.get('role') == 'user':
                    paper_text = message.get('content', "")
                    break

        example['paper_text'] = paper_text
        return example

    # Apply extraction to test split
    ds = ds.map(extract_paper_text)

    # 3. Append all paper_text to a list
    print("Collecting all paper texts...")
    paper_texts = list(ds['paper_text'])

    # Randomly sample papers with seed for reproducibility
    if len(paper_texts) > NUM_SAMPLES:
        # Get random indices
        sampled_indices = random.sample(range(len(paper_texts)), NUM_SAMPLES)
        sampled_indices.sort()  # Sort to maintain some order
        paper_texts = [paper_texts[i] for i in sampled_indices]
        # Also need to filter the dataset to match
        ds = ds.select(sampled_indices)

    print(f"Sampled {len(paper_texts)} papers (seed: {RANDOM_SEED})")

    # 4. Initialize DeepReviewer and run evaluations
    print(f"\nInitializing DeepReviewer with model: {args.model_name}...")
    deep_reviewer = DeepReviewer(model_name=args.model_name, tensor_parallel_size=1)

    # Start timing
    start_time = time.time()
    print(f"\nRunning {args.mode} evaluation...")
    review_results = deep_reviewer.evaluate(
        paper_texts,
        mode=args.mode,  # Options: "base_mode", "fast_mode", "standard_mode", "best_mode"
        reviewer_num=4         # Simulate 4 different reviewers
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\n{args.mode} evaluation completed in {elapsed_time:.2f} seconds")
    
    # print("\nRunning standard_mode evaluation...")
    # standard_review_results = deep_reviewer.evaluate(
    #     paper_texts,
    #     mode="standard_mode",  # Options: "fast_mode", "standard_mode", "best_mode"
    #     reviewer_num=4         # Simulate 4 different reviewers
    # )

    print("\nBenchmark completed!")
    print(f"{args.mode} results: {len(review_results)} reviews")

    # 5. Prepare output data with all required fields
    print("\nPreparing output data...")
    output_data = []

    
    for i in range(len(paper_texts)):
        entry = {
            'id': ds[i].get('id', i),
            'title': ds[i].get('title', ''),
            'paper_context': ds[i].get('paper_text', ''),
            'decision': ds[i].get('decision', ''),
            'review': ds[i].get('reviewer_comments', ''),
            f'pred_{args.mode}': review_results[i],
        }
        output_data.append(entry)

    # 6. Save to JSON file
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    # Extract model name for filename (e.g., "WestlakeNLP/DeepReviewer-7B" -> "DeepReviewer-7B")
    model_filename = args.model_name.split('/')[-1]
    output_dir = 'evaluate/review'
    os.makedirs(output_dir, exist_ok=True)
    output_file = f'{output_dir}/deepreviewer_{model_filename}_{timestamp}.json'
    print(f"\nSaving results to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"Results saved successfully! Total entries: {len(output_data)}")

    return ds, paper_texts, review_results, output_data


if __name__ == "__main__":
    args = parse_args()
    dataset, papers, review_results, output_data = main(args)

"""
Example usage:
python benchmark_deepreviewer.py --model-name WestlakeNLP/DeepReviewer-7B --mode fast_mode
python benchmark_deepreviewer.py --model-name WestlakeNLP/DeepReviewer-7B --mode standard_mode
python benchmark_deepreviewer.py --model-name Qwen/Qwen3-4B-Instruct-2507 --mode base_mode
python benchmark_deepreviewer.py --model-name Qwen/Qwen3-8B --mode base_mode
"""

# 1. update prompt 
# 2. use qwen3-4b-non-thinking model 