from datasets import load_dataset
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Load the deepreview-sft dataset
print("Loading dataset...")
ds = load_dataset("ZhuofengLi/deepreview-sft-fast")

# Initialize tokenizer (using a common model tokenizer)
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")

# Count tokens for each conversation
print("Processing conversations and counting tokens...")
token_counts = []

for split in ds.keys():
    print(f"Processing split: {split}")
    for example in tqdm(ds[split], desc=f"Processing {split}"):
        # Get the conversations
        conversations = example.get('conversations', example.get('messages', []))

        # Apply chat template
        if conversations:
            try:
                # Apply the tokenizer's chat template
                formatted_text = tokenizer.apply_chat_template(
                    conversations,
                    tokenize=False,
                    add_generation_prompt=False
                )

                # Count tokens
                tokens = tokenizer.encode(formatted_text)
                token_count = len(tokens)
                token_counts.append(token_count)
            except Exception as e:
                print(f"Error processing conversation: {e}")
                continue

print(f"\nTotal conversations processed: {len(token_counts)}")
print(f"Min tokens: {min(token_counts)}")
print(f"Max tokens: {max(token_counts)}")
print(f"Mean tokens: {np.mean(token_counts):.2f}")
print(f"Median tokens: {np.median(token_counts):.2f}")

# Create histogram
plt.figure(figsize=(12, 6))
plt.hist(token_counts, bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Token Count')
plt.ylabel('Frequency')
plt.title('Distribution of Token Counts in DeepReview-SFT Dataset')
plt.grid(True, alpha=0.3)

# Add statistics to the plot
stats_text = f'Total: {len(token_counts)}\nMin: {min(token_counts)}\nMax: {max(token_counts)}\nMean: {np.mean(token_counts):.2f}\nMedian: {np.median(token_counts):.2f}'
plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes,
         verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Save the plot
output_file = 'token_distribution.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\nHistogram saved to: {output_file}")

# Also save the raw data
import json
with open('token_counts.json', 'w') as f:
    json.dump({
        'token_counts': token_counts,
        'statistics': {
            'total': len(token_counts),
            'min': int(min(token_counts)),
            'max': int(max(token_counts)),
            'mean': float(np.mean(token_counts)),
            'median': float(np.median(token_counts)),
            'std': float(np.std(token_counts))
        }
    }, f, indent=2)
print("Token counts data saved to: token_counts.json")

plt.show()
