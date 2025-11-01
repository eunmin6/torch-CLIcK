from huggingface_hub import hf_hub_download
import json
import os

# Download a sample file to analyze the data structure
sample_files = [
    "Dataset/Culture/Korean History/History_KHB.json",
    "Dataset/Language/Grammar/Grammar_CSAT.json"
]

print("=== Downloading sample files ===")
for file_path in sample_files:
    try:
        local_path = hf_hub_download(
            repo_id="EunsuKim/CLIcK",
            filename=file_path,
            repo_type="dataset"
        )
        print(f"\nAnalyzing: {file_path}")
        print(f"Downloaded to: {local_path}")

        with open(local_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"Number of examples: {len(data)}")

        if len(data) > 0:
            print("\nFirst example:")
            first_item = data[0]
            for key, value in first_item.items():
                if isinstance(value, str) and len(value) > 100:
                    print(f"  {key}: {value[:100]}... (length: {len(value)})")
                else:
                    print(f"  {key}: {value}")

            print("\nKeys in examples:")
            print(list(first_item.keys()))

            # Show 2 more examples
            if len(data) >= 3:
                print("\n--- Additional samples ---")
                for i in range(1, min(3, len(data))):
                    print(f"\nExample {i+1}:")
                    for key, value in data[i].items():
                        if isinstance(value, str) and len(value) > 80:
                            print(f"  {key}: {value[:80]}...")
                        else:
                            print(f"  {key}: {value}")

        print("\n" + "="*80)

    except Exception as e:
        print(f"Error loading {file_path}: {e}")
