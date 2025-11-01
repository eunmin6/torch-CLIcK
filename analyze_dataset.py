from datasets import load_dataset

print("Loading CLIcK dataset...")
ds = load_dataset("EunsuKim/CLIcK")

print("\n=== Dataset Structure ===")
print(ds)

print("\n=== Dataset Splits ===")
print(list(ds.keys()))

print("\n=== Dataset Sizes ===")
for split in ds.keys():
    print(f"{split}: {len(ds[split])} examples")

# Get the first split available
first_split = list(ds.keys())[0]
print(f"\n=== Features (from {first_split} split) ===")
print(ds[first_split].features)

print(f"\n=== First Example (from {first_split} split) ===")
first_example = ds[first_split][0]
for key, value in first_example.items():
    if isinstance(value, str) and len(value) > 200:
        print(f"{key}: {value[:200]}... (truncated)")
    else:
        print(f"{key}: {value}")

print(f"\n=== Sample of 3 Examples ===")
for i in range(min(3, len(ds[first_split]))):
    print(f"\nExample {i+1}:")
    example = ds[first_split][i]
    for key, value in example.items():
        if isinstance(value, str) and len(value) > 100:
            print(f"  {key}: {value[:100]}... (length: {len(value)})")
        else:
            print(f"  {key}: {value}")
