from datasets import load_dataset, Dataset, DatasetDict
from tqdm import tqdm
import json

# Load the original dataset
print("Loading original dataset...")
ds = load_dataset("ZhuofengLi/deepreview-sft")

# Filter data where system role contains the target string
print("Filtering data where system role contains 'Your thinking mode is Fast Mode. In this mode,'...")
train_filtered = []
test_filtered = []

for split in ds.keys():
    print(f"\nProcessing split: {split}")
    for example in tqdm(ds[split], desc=f"Processing {split}"):
        # Get the conversations
        conversations = example.get('conversations', example.get('messages', []))

        # Check if any system message contains the target string
        has_fast_mode = False
        for msg in conversations:
            role = msg.get('role') or msg.get('from')
            content = msg.get('content') or msg.get('value', '')

            if role == 'system' and 'Your thinking mode is Fast Mode. In this mode,' in content:
                has_fast_mode = True
                break

        if has_fast_mode:
            # Keep the original structure
            if split == 'train':
                train_filtered.append(example)
            elif split == 'test':
                test_filtered.append(example)

print(f"\nFiltered train examples: {len(train_filtered)}")
print(f"Filtered test examples: {len(test_filtered)}")

# Create Dataset objects
print("\nCreating Dataset objects...")

# If test is empty, create a train/test split from train data
if len(test_filtered) == 0 and len(train_filtered) > 0:
    print("\nTest split is empty. Creating train/test split from train data...")
    # Use 95% for train, 5% for test
    split_idx = int(len(train_filtered) * 0.95)
    train_data = train_filtered[:split_idx]
    test_data = train_filtered[split_idx:]

    print(f"New train examples: {len(train_data)}")
    print(f"New test examples: {len(test_data)}")

    train_dataset = Dataset.from_list(train_data)
    test_dataset = Dataset.from_list(test_data)
else:
    train_dataset = Dataset.from_list(train_filtered)
    test_dataset = Dataset.from_list(test_filtered)

# Create DatasetDict
dataset_dict = DatasetDict({
    'train': train_dataset,
    'test': test_dataset
})

print("\nDataset structure:")
print(dataset_dict)

# Push to Hugging Face Hub
print("\nPushing to Hugging Face Hub...")
print("Repository: ZhuofengLi/deepreview-sft-fast")

try:
    dataset_dict.push_to_hub(
        "ZhuofengLi/deepreview-sft-fast",
        private=False,  # Set to True if you want a private dataset
        token=None  # Will use the cached token from huggingface-cli login
    )
    print("\n✓ Dataset successfully pushed to Hugging Face Hub!")
    print("URL: https://huggingface.co/datasets/ZhuofengLi/deepreview-sft-fast")
except Exception as e:
    print(f"\n✗ Error pushing to Hugging Face Hub: {e}")
    print("\nMake sure you are logged in with: huggingface-cli login")
    print("Or set your token with: export HF_TOKEN=your_token_here")
