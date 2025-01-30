from datasets import load_dataset

# Load the 'dair-ai/emotion' dataset
dataset = load_dataset('dair-ai/emotion')

# Initialize counters
total_words = 0
total_characters = 0
vocabulary = set()
total_examples = 0

# Iterate through the dataset to compute statistics
for split in dataset.keys():
    for example in dataset[split]:
        text = example['text']
        words = text.split()
        total_words += len(words)
        total_characters += len(text)
        vocabulary.update(words)
        total_examples += 1

# Compute required statistics
average_words_per_example = total_words / total_examples
average_characters_per_example = total_characters / total_examples
vocabulary_size = len(vocabulary)

# Print results
print(f"Average number of words per example (W): {average_words_per_example:.2f}")
print(f"Average number of characters per example (L): {average_characters_per_example:.2f}")
print(f"Vocabulary size (V): {vocabulary_size}")
