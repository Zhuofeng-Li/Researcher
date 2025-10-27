"""
Benchmark script for DeepReviewer using WestlakeNLP/DeepReview-13K dataset
"""

import json
import os
import random
from datasets import load_dataset
from ai_researcher import DeepReviewer
import time
from datetime import datetime
# Set CUDA device if needed
# os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

# Configuration
RANDOM_SEED = 42
NUM_SAMPLES = 100

# Set random seed for reproducibility
random.seed(RANDOM_SEED)


def main():

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
    print("\nInitializing DeepReviewer...")
    deep_reviewer = DeepReviewer(model_size="7B", tensor_parallel_size=2)

    # Start timing
    start_time = time.time()
    print("\nRunning Fast Mode evaluation...")
    fast_review_results = deep_reviewer.evaluate(
        paper_texts,
        mode="Fast Mode",  # Options: "Fast Mode", "Standard Mode", "Best Mode"
        reviewer_num=4         # Simulate 4 different reviewers
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nFast Mode evaluation completed in {elapsed_time:.2f} seconds")
    
    # print("\nRunning Standard Mode evaluation...")
    # standard_review_results = deep_reviewer.evaluate(
    #     paper_texts,
    #     mode="Standard Mode",  # Options: "Fast Mode", "Standard Mode", "Best Mode"
    #     reviewer_num=4         # Simulate 4 different reviewers
    # )

    print("\nBenchmark completed!")
    print(f"Fast Mode results: {len(fast_review_results)} reviews")
    # print(f"Standard Mode results: {len(standard_review_results)} reviews")

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
            'pred_fast_mode': fast_review_results[i],
            'pred_standard_mode': '',  # standard_review_results[i],
        }
        output_data.append(entry)

    # 6. Save to JSON file
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_file = f'evaluate/review/benchmark_deepreviewer_results_{timestamp}.json'
    print(f"\nSaving results to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"Results saved successfully! Total entries: {len(output_data)}")

    return ds, paper_texts, fast_review_results, output_data


if __name__ == "__main__":
    dataset, papers, fast_results, output_data = main()
