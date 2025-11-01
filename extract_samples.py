"""
샘플 예측 결과 추출 스크립트
"""
import json
import random

# Load all predictions
predictions = []
with open('./benchmarks/baseline/predictions.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        predictions.append(json.loads(line))

# Separate correct and incorrect
correct = [p for p in predictions if p.get('is_correct', False)]
incorrect = [p for p in predictions if not p.get('is_correct', False)]

print(f"Total predictions: {len(predictions)}")
print(f"Correct: {len(correct)}")
print(f"Incorrect: {len(incorrect)}")

# Sample 5 from each
random.seed(42)
sample_correct = random.sample(correct, min(5, len(correct)))
sample_incorrect = random.sample(incorrect, min(5, len(incorrect)))

# Combine and shuffle
samples = sample_correct + sample_incorrect
random.shuffle(samples)

# Save to JSON
with open('samples.json', 'w', encoding='utf-8') as f:
    json.dump(samples, f, ensure_ascii=False, indent=2)

print(f"\nExtracted {len(samples)} samples to samples.json")
