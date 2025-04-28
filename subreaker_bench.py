import os
import time
import json
import random
import string
import argparse
import multiprocessing
from pathlib import Path
import subbreaker
from concurrent.futures import ProcessPoolExecutor

class CipherGenerator:
    # Keeping your existing CipherGenerator class unchanged
    def __init__(self, alphabet="abcdefghijklmnopqrstuvwxyz", handle_special_chars="preserve"):
        self.alphabet = alphabet
        self.handle_special_chars = handle_special_chars
    
    def generate_random_key(self):
        """Generate a random substitution cipher key"""
        key_chars = list(self.alphabet)
        random.shuffle(key_chars)
        return ''.join(key_chars)
    
    def preprocess_text(self, text):
        """Preprocess text based on special character handling strategy"""
        if self.handle_special_chars == "remove":
            return ''.join(c for c in text.lower() if c in self.alphabet)
        elif self.handle_special_chars == "include":
            if any(c not in self.alphabet for c in text.lower() if c.strip()):
                print("Warning: Some characters in text are not in alphabet")
            return text
        else:  # "preserve" - default
            return text
    
    def encrypt_text(self, plaintext, key=None):
        """Encrypt plaintext using a random or provided key"""
        if not key:
            key = self.generate_random_key()
        
        processed_text = self.preprocess_text(plaintext)
        cipher = subbreaker.Key(key, self.alphabet)
        ciphertext = cipher.encode(processed_text)
        return ciphertext, key

def break_cipher_batch(task_args):
    """Process multiple ciphers in one process to reduce overhead"""
    quadgram_file, ciphertexts, task_ids, max_rounds, consolidate = task_args
    
    start_time = time.time()
    
    # Initialize breaker only ONCE per batch - this is key to performance
    with open(quadgram_file, 'r') as f:
        breaker = subbreaker.Breaker(f)
    
    results = []
    
    # Process all ciphers in the batch with the same breaker instance
    for i, (ciphertext, task_id) in enumerate(zip(ciphertexts, task_ids)):
        cipher_start = time.time()
        
        # Measure original fitness
        original_fitness = breaker.calc_fitness(ciphertext)
        
        try:
            # Break the cipher
            result = breaker.break_cipher(
                ciphertext,
                max_rounds=max_rounds,
                consolidate=consolidate
            )
            
            elapsed_time = time.time() - cipher_start
            keys_per_second = result.keys_per_second
            
        except ZeroDivisionError:
            # Handle the case where PyPy is too fast (time measured as 0)
            elapsed_time = time.time() - cipher_start
            
            # If elapsed time is still 0, use a small value
            if elapsed_time <= 0:
                elapsed_time = 0.001  # 1ms minimum to avoid division by zero
            
            # Calculate keys per second manually
            keys_per_second = result.nbr_keys / elapsed_time
            
            # Update the result with corrected keys per second
            # Note: Since we can't modify the original result object easily,
            # we'll just use the corrected value when building our results dict
        
        results.append({
            "elapsed_time": elapsed_time,
            "keys_tried": result.nbr_keys,
            "keys_per_second": keys_per_second,
            "original_fitness": original_fitness,
            "final_fitness": result.fitness,
            "success": result.fitness > 90,
            "task_id": task_id
        })
    
    return results

def run_benchmark(args):
    # Create output directory if needed
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load or create sample text
    if args.text_file and os.path.exists(args.text_file):
        with open(args.text_file, 'r') as f:
            sample_text = f.read()
    else:
        # Use a default sample text if no file provided
        sample_text = """The quick brown fox jumps over the lazy dog. 
        Pack my box with five dozen liquor jugs.
        How vexingly quick daft zebras jump!"""
        
        # Make it longer if needed
        if len(sample_text) < 500:
            sample_text = sample_text * (500 // len(sample_text) + 1)
    
    # Generate ciphers
    generator = CipherGenerator(
        args.alphabet,
        handle_special_chars=args.handle_special_chars
    )
    
    print(f"Generating {args.num_ciphers} ciphers...")
    ciphers = []
    original_keys = []
    
    for i in range(args.num_ciphers):
        key = generator.generate_random_key()
        ciphertext, _ = generator.encrypt_text(sample_text, key)
        ciphers.append(ciphertext)
        original_keys.append(key)
    
    # Divide ciphers into batches based on process count
    batch_size = max(1, args.num_ciphers // args.processes)
    batches = []
    
    for i in range(0, args.num_ciphers, batch_size):
        end_idx = min(i + batch_size, args.num_ciphers)
        batch_ciphers = ciphers[i:end_idx]
        batch_ids = list(range(i, end_idx))
        batches.append((args.quadgrams, batch_ciphers, batch_ids, args.max_rounds, args.consolidate))
    
    print(f"Breaking {args.num_ciphers} ciphers using {args.processes} processes")
    print(f"Batch size: ~{batch_size} ciphers per process")
    
    # Process batches
    start_time = time.time()
    all_results = []
    
    with ProcessPoolExecutor(max_workers=args.processes) as executor:
        for batch_results in executor.map(break_cipher_batch, batches):
            all_results.extend(batch_results)
            # Print progress for each completed task
            #for result in batch_results:
            #    print(f"Completed task {result['task_id']+1}/{args.num_ciphers}, "
            #          f"time: {result['elapsed_time']:.2f}s, "
            #          f"keys/s: {result['keys_per_second']:.2f}")
    
    total_time = time.time() - start_time
    
    # Calculate aggregate statistics
    successful_tasks = [r for r in all_results if r["success"]]
    success_rate = len(successful_tasks) / len(all_results) if all_results else 0
    avg_keys_per_second = sum(r["keys_per_second"] for r in all_results) / len(all_results) if all_results else 0
    total_keys_tried = sum(r["keys_tried"] for r in all_results)
    
    # Prepare benchmark results
    benchmark_results = {
        "benchmark_parameters": {
            "processes": args.processes,
            "num_ciphers": args.num_ciphers,
            "max_rounds": args.max_rounds,
            "consolidate": args.consolidate,
            "alphabet": args.alphabet,
            "batch_size": batch_size
        },
        "performance_metrics": {
            "total_time": total_time,
            "success_rate": success_rate,
            "avg_keys_per_second": avg_keys_per_second,
            "total_keys_tried": total_keys_tried,
            "keys_per_second_per_process": avg_keys_per_second / args.processes
        },
        "detailed_results": all_results
    }
    
    # Save results
    output_file = os.path.join(args.output_dir, f"benchmark_batch_p{args.processes}_c{args.num_ciphers}.json")
    with open(output_file, 'w') as f:
        json.dump(benchmark_results, f, indent=2)
    
    print("\nBenchmark Results:")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Success rate: {success_rate*100:.2f}%")
    print(f"Average keys/second: {avg_keys_per_second:.2f}")
    print(f"Keys/second/process: {avg_keys_per_second/args.processes:.2f}")
    print(f"Total keys tried: {total_keys_tried}")
    print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    # Make sure multiprocessing works on Windows
    multiprocessing.freeze_support()
    
    parser = argparse.ArgumentParser(description="Batch-optimized subbreaker benchmark")
    parser.add_argument("--processes", type=int, default=os.cpu_count(),
                        help="Number of processes to use (default: number of CPU cores)")
    parser.add_argument("--num-ciphers", type=int, default=10,
                        help="Number of ciphers to generate and break")
    parser.add_argument("--max-rounds", type=int, default=10000,
                        help="Maximum number of rounds for breaking each cipher")
    parser.add_argument("--consolidate", type=int, default=3,
                        help="Consolidation parameter for subbreaker")
    parser.add_argument("--text-file", type=str, default=None,
                        help="Path to sample text file (optional)")
    parser.add_argument("--quadgrams", type=str, required=True,
                        help="Path to the quadgrams file")
    parser.add_argument("--output-dir", type=str, default="benchmark_results",
                        help="Output directory for benchmark results")
    parser.add_argument("--alphabet", type=str, default="abcdefghijklmnopqrstuvwxyz",
                        help="Alphabet to use for cipher generation")
    parser.add_argument("--handle-special-chars", type=str, default="preserve",
                        choices=["preserve", "remove", "include"],
                        help="How to handle characters not in the alphabet")
    
    args = parser.parse_args()
    run_benchmark(args)