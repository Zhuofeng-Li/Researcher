from datasets import load_dataset
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import re

def extract_boxed_review(text):
    """
    Extract content from \\boxed_review{...} pattern.
    Handles nested braces properly.
    """
    # Find the start of \\boxed_review{
    pattern = r'\\boxed_review\s*\{'
    match = re.search(pattern, text)

    if not match:
        return None

    # Start position after the opening brace
    start = match.end()

    # Count braces to find the matching closing brace
    brace_count = 1
    pos = start

    while pos < len(text) and brace_count > 0:
        if text[pos] == '{':
            brace_count += 1
        elif text[pos] == '}':
            brace_count -= 1
        pos += 1

    if brace_count == 0:
        # Found matching brace
        return text[start:pos-1]
    else:
        # Unmatched braces
        return None

# Load the deepreview-sft dataset
print("Loading dataset...")
ds = load_dataset("ZhuofengLi/deepreview-sft")

# Initialize tokenizer (using a common model tokenizer)
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")

# Count tokens for \\boxed_review content in last assistant response
print("Processing conversations and counting tokens for \\boxed_review content...")
token_counts = []
no_boxed_review_count = 0

for split in ds.keys():
    print(f"Processing split: {split}")
    for example in tqdm(ds[split], desc=f"Processing {split}"):
        # Get the conversations
        conversations = example.get('conversations', example.get('messages', []))

        # Find the last assistant message
        if conversations:
            try:
                # Find last message with role "assistant"
                last_assistant_msg = None
                for msg in reversed(conversations):
                    if msg.get('role') == 'assistant' or msg.get('from') == 'assistant':
                        last_assistant_msg = msg
                        break

                if last_assistant_msg:
                    # Get the content of the last assistant message
                    content = last_assistant_msg.get('content', last_assistant_msg.get('value', ''))

                    # Extract \\boxed_review content
                    boxed_content = extract_boxed_review(content)

                    if boxed_content:
                        # Count tokens for the boxed content only
                        tokens = tokenizer.encode(boxed_content)
                        token_count = len(tokens)
                        token_counts.append(token_count)
                    else:
                        no_boxed_review_count += 1
                else:
                    print(f"Warning: No assistant message found in conversation")
            except Exception as e:
                print(f"Error processing conversation: {e}")
                continue

print(f"\nTotal conversations with \\boxed_review: {len(token_counts)}")
print(f"Conversations without \\boxed_review: {no_boxed_review_count}")
print(f"Min tokens: {min(token_counts)}")
print(f"Max tokens: {max(token_counts)}")
print(f"Mean tokens: {np.mean(token_counts):.2f}")
print(f"Median tokens: {np.median(token_counts):.2f}")

# Create histogram
plt.figure(figsize=(12, 6))
plt.hist(token_counts, bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Token Count')
plt.ylabel('Frequency')
plt.title('Distribution of Token Counts in \\boxed_review Content (DeepReview-SFT Dataset)')
plt.grid(True, alpha=0.3)

# Add statistics to the plot
stats_text = f'With \\boxed_review: {len(token_counts)}\nWithout: {no_boxed_review_count}\nMin: {min(token_counts)}\nMax: {max(token_counts)}\nMean: {np.mean(token_counts):.2f}\nMedian: {np.median(token_counts):.2f}'
plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes,
         verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Save the plot
output_file = 'boxed_review_token_distribution.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\nHistogram saved to: {output_file}")

# Also save the raw data
import json
with open('boxed_review_token_counts.json', 'w') as f:
    json.dump({
        'token_counts': token_counts,
        'statistics': {
            'total_with_boxed_review': len(token_counts),
            'total_without_boxed_review': no_boxed_review_count,
            'min': int(min(token_counts)),
            'max': int(max(token_counts)),
            'mean': float(np.mean(token_counts)),
            'median': float(np.median(token_counts)),
            'std': float(np.std(token_counts))
        }
    }, f, indent=2)
print("Token counts data saved to: boxed_review_token_counts.json")

plt.show()
