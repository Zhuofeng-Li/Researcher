from datasets import load_dataset
from tqdm import tqdm
import json

# Load the deepreview-sft dataset
print("Loading dataset...")
ds = load_dataset("ZhuofengLi/deepreview-sft")

# Filter data where system role contains the target string
print("Filtering data where system role contains 'Your thinking mode is Fast Mode. In this mode,'...")
filtered_data = []
total_count = 0
matched_count = 0

for split in ds.keys():
    print(f"\nProcessing split: {split}")
    for example in tqdm(ds[split], desc=f"Processing {split}"):
        total_count += 1
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
            matched_count += 1
            filtered_data.append({
                'split': split,
                'conversations': conversations,
                'original_example': example
            })

print(f"\n{'='*60}")
print(f"Total examples processed: {total_count}")
print(f"Examples with 'Fast Mode' in system role: {matched_count}")
print(f"Percentage: {(matched_count/total_count)*100:.2f}%")
print(f"{'='*60}")

# Save filtered data to JSON file
output_file = 'filtered_fast_mode_data.json'
print(f"\nSaving filtered data to: {output_file}")
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(filtered_data, f, indent=2, ensure_ascii=False)

print(f"Filtered data saved successfully!")

# Also save a summary with just the conversations
output_file_simple = 'filtered_fast_mode_conversations.json'
print(f"Saving simplified version to: {output_file_simple}")
simple_data = [
    {
        'split': item['split'],
        'conversations': item['conversations']
    }
    for item in filtered_data
]
with open(output_file_simple, 'w', encoding='utf-8') as f:
    json.dump(simple_data, f, indent=2, ensure_ascii=False)

print("Done!")

# Print a sample of the first filtered example (if any)
if filtered_data:
    print(f"\n{'='*60}")
    print("Sample of first filtered example:")
    print(f"{'='*60}")
    for i, msg in enumerate(filtered_data[0]['conversations'][:3]):  # Show first 3 messages
        role = msg.get('role') or msg.get('from')
        content = msg.get('content') or msg.get('value', '')
        print(f"\nMessage {i+1} - Role: {role}")
        print(f"Content preview: {content[:200]}...")
