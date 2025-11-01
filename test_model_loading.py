"""
Qwen2.5-7B-Instruct 모델 로드 테스트

이 스크립트는:
1. 모델과 토크나이저 로드 (BF16, quantization 비활성화)
2. Chat template 확인
3. 간단한 추론 테스트
4. 메모리 사용량 측정

Note: gpt-oss-20b에서 변경됨 (GPU 메모리 제약으로 인해)
Note2: Llama-3.1에서 변경됨 (gated model 인증 이슈)
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import psutil
import os

# PyTorch CUDA 메모리 관리 설정 (fragmentation 방지)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

def get_memory_usage():
    """현재 GPU 메모리 사용량 확인"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3  # GB
        return allocated, reserved
    return 0, 0

def main():
    print("="*80)
    print("[TEST] Qwen2.5-7B-Instruct Model Loading Test")
    print("="*80 + "\n")

    # GPU 캐시 정리
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("[INFO] GPU cache cleared")

    # GPU 확인
    print("[INFO] GPU Information:")
    if torch.cuda.is_available():
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        print(f"  Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("  [WARN] CUDA not available!")
        return

    # 토크나이저 로드
    print("\n[INFO] Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        print("[OK] Tokenizer loaded successfully")

        # Chat template 확인
        if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
            print("[OK] Chat template is available")
        else:
            print("[WARN] Chat template not found")
    except Exception as e:
        print(f"[ERROR] Failed to load tokenizer: {e}")
        return

    # 모델 로드 (quantization 비활성화)
    print("\n[INFO] Loading model (this may take a few minutes)...")
    print("[INFO] Using BF16 without quantization (Python 3.14 compatibility)...")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",  # Force all layers to GPU
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        print("[OK] Model loaded successfully")

        # 메모리 사용량 확인
        allocated, reserved = get_memory_usage()
        print(f"\n[INFO] GPU Memory Usage:")
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Reserved: {reserved:.2f} GB")

    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return

    # 간단한 추론 테스트
    print("\n[INFO] Testing inference with chat template...")

    test_messages = [
        {"role": "user", "content": "안녕하세요! 한국의 수도는 어디인가요?"}
    ]

    try:
        # Chat template 적용
        input_text = tokenizer.apply_chat_template(
            test_messages,
            tokenize=False,
            add_generation_prompt=True
        )

        print("[OK] Chat template applied")
        print(f"\n[INFO] Formatted input (first 200 chars):")
        print(f"  {input_text[:200]}...")

        # 토크나이징
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        print(f"\n[INFO] Input length: {inputs['input_ids'].shape[1]} tokens")

        # 생성
        print("\n[INFO] Generating response...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        # 디코딩
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print("[OK] Generation complete")
        print(f"\n[INFO] Generated text (first 500 chars):")
        print(f"  {generated_text[:500]}...")

        # 최종 메모리 사용량
        allocated, reserved = get_memory_usage()
        print(f"\n[INFO] GPU Memory Usage After Generation:")
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Reserved: {reserved:.2f} GB")

    except Exception as e:
        print(f"[ERROR] Inference test failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "="*80)
    print("[SUCCESS] Model loading and inference test completed!")
    print("="*80)

if __name__ == "__main__":
    main()
