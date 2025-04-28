import os
import time
import json
import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import subbreaker

def find_grid_sizes(text_length):
    """Find all possible grid sizes for columnar transposition"""
    grid_sizes = []
    
    # Try all factors where both dimensions are at least 3
    for rows in range(3, int(text_length**0.5) + 1):
        if text_length % rows == 0:
            cols = text_length // rows
            if cols >= 3:  # Ensure minimum dimension
                grid_sizes.append((rows, cols))
    
    # Also add the transposed dimensions (rows and columns swapped)
    for size in list(grid_sizes):  # Use list() to avoid modifying during iteration
        rows, cols = size
        if rows != cols and (cols, rows) not in grid_sizes:
            grid_sizes.append((cols, rows))
    
    return sorted(grid_sizes)

def reverse_columnar_transposition(ciphertext, rows, cols):
    """Reverse a columnar transposition with given dimensions"""
    # Create an empty grid
    grid = [[''] * cols for _ in range(rows)]
    
    # Fill the grid by columns (as it was read out during encryption)
    index = 0
    for col in range(cols):
        for row in range(rows):
            if index < len(ciphertext):
                grid[row][col] = ciphertext[index]
                index += 1
    
    # Read out by rows to get the original text before substitution
    result = ''
    for row in range(rows):
        result += ''.join(grid[row])
    
    return result

def reverse_double_transposition(ciphertext, first_rows, first_cols, second_rows, second_cols):
    """Reverse a double columnar transposition with given dimensions"""
    # First, reverse the second transposition
    intermediate = reverse_columnar_transposition(ciphertext, second_rows, second_cols)
    
    # Then reverse the first transposition
    result = reverse_columnar_transposition(intermediate, first_rows, first_cols)
    
    return result

def break_substitution(task_args):
    """Break the substitution cipher on a transposed text"""
    untransposed, quadgram_file, first_rows, first_cols, second_rows, second_cols, max_rounds, consolidate = task_args
    
    # Initialize the breaker
    start_time = time.time()
    with open(quadgram_file, 'r') as f:
        breaker = subbreaker.Breaker(f)
    
    # Break the substitution cipher
    result = breaker.break_cipher(
        untransposed,
        max_rounds=max_rounds,
        consolidate=consolidate
    )
    
    elapsed_time = time.time() - start_time
    
    # Create a Key object from the result key string
    key_obj = subbreaker.Key(result.key, alphabet=result.alphabet)
    
    # Decode the text using the key object
    decrypted = key_obj.decode(untransposed)
    
    return {
        "first_grid": (first_rows, first_cols),
        "second_grid": (second_rows, second_cols),
        "untransposed": untransposed,
        "decrypted": decrypted,
        "key": result.key,  # String representation of the key
        "fitness": result.fitness,
        "keys_tried": result.nbr_keys,
        "elapsed_time": elapsed_time
    }

def break_double_transposition_cipher(ciphertext, quadgram_file, max_rounds=10000, consolidate=3, processes=None):
    """Break a double columnar transposition followed by substitution cipher"""
    if not processes:
        processes = max(1, multiprocessing.cpu_count() - 1)  # Leave one CPU free
    
    # Get all possible grid sizes for both transpositions
    text_length = len(ciphertext)
    first_grid_sizes = find_grid_sizes(text_length)
    
    if not first_grid_sizes:
        print("No valid grid sizes found for this text length.")
        return []
    
    print(f"Found {len(first_grid_sizes)} possible grid sizes for first transposition:")
    for i, (rows, cols) in enumerate(first_grid_sizes):
        print(f"{i+1}. {rows} rows × {cols} columns")
    
    # Prepare tasks for parallel processing
    tasks = []
    grid_combinations = []
    
    # For each possible first transposition grid
    for first_rows, first_cols in first_grid_sizes:
        # For each first transposition, get possible second transposition grids
        intermediate_length = text_length  # Length remains the same after transposition
        second_grid_sizes = find_grid_sizes(intermediate_length)
        
        for second_rows, second_cols in second_grid_sizes:
            # Reverse both transpositions
            untransposed = reverse_double_transposition(
                ciphertext, 
                first_rows, first_cols, 
                second_rows, second_cols
            )
            
            # Create task for breaking substitution
            tasks.append((
                untransposed, 
                quadgram_file, 
                first_rows, first_cols, 
                second_rows, second_cols, 
                max_rounds, 
                consolidate
            ))
            
            grid_combinations.append(((first_rows, first_cols), (second_rows, second_cols)))
    
    print(f"Testing {len(tasks)} combinations of double transposition grids")
    
    # Process tasks in parallel
    results = []
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=processes) as executor:
        for i, result in enumerate(executor.map(break_substitution, tasks)):
            results.append(result)
            
            first_grid = result["first_grid"]
            second_grid = result["second_grid"]
            print(f"\nCompleted {i+1}/{len(tasks)}: First grid {first_grid[0]}×{first_grid[1]}, Second grid {second_grid[0]}×{second_grid[1]}")
            print(f"Fitness: {result['fitness']:.2f}")
            print(f"Keys tried: {result['keys_tried']}")
            print(f"Time: {result['elapsed_time']:.2f}s")
    
    total_time = time.time() - start_time
    print(f"\nTotal processing time: {total_time:.2f}s")
    
    # Sort results by fitness
    results.sort(key=lambda x: x["fitness"], reverse=True)
    
    return results

def get_caesar_key(shift, alphabet="abcdefghijklmnopqrstuvwxyz"):
    """Generate a substitution key for a Caesar cipher with given shift."""
    return alphabet[shift:] + alphabet[:shift]

def try_caesar_shifts(untransposed, breaker, fitness_threshold=90):
    """Try all 26 Caesar shifts for a given text and return those above threshold."""
    results = []
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    
    # Try all 26 possible shifts
    for shift in range(26):
        # Generate Caesar key
        caesar_key = get_caesar_key(shift)
        key_obj = subbreaker.Key(caesar_key)
        
        # Decode with this key
        decrypted = key_obj.decode(untransposed)
        
        # Calculate fitness
        fitness = breaker.calc_fitness(decrypted)
        
        # If fitness is above threshold, save this result
        if fitness >= fitness_threshold:
            results.append({
                "shift": shift,
                "key": caesar_key,
                "decrypted": decrypted,
                "fitness": fitness
            })
    
    return results

def break_caesar_single_transposition(task_args):
    """Process a batch of single transpositions with Caesar cipher."""
    ciphertext_batch, quadgram_file, grid_sizes, fitness_threshold = task_args
    
    # Initialize breaker once for this batch
    with open(quadgram_file, 'r') as f:
        breaker = subbreaker.Breaker(f)
    
    all_results = []
    
    # For each grid size in this batch
    for rows, cols in grid_sizes:
        # Reverse the transposition
        untransposed = reverse_columnar_transposition(ciphertext_batch, rows, cols)
        
        # Try all Caesar shifts and get results above threshold
        caesar_results = try_caesar_shifts(untransposed, breaker, fitness_threshold)
        
        # Add grid info to each result
        for result in caesar_results:
            result["rows"] = rows
            result["cols"] = cols
            all_results.append(result)
    
    return all_results

def break_caesar_double_transposition(task_args):
    """Process a batch of double transpositions with Caesar cipher."""
    ciphertext_batch, quadgram_file, grid_combinations, fitness_threshold = task_args
    
    # Initialize breaker once for this batch
    with open(quadgram_file, 'r') as f:
        breaker = subbreaker.Breaker(f)
    
    all_results = []
    
    # For each grid combination in this batch
    for first_grid, second_grid in grid_combinations:
        first_rows, first_cols = first_grid
        second_rows, second_cols = second_grid
        
        # Reverse the double transposition
        untransposed = reverse_double_transposition(
            ciphertext_batch,
            first_rows, first_cols,
            second_rows, second_cols
        )
        
        # Try all Caesar shifts and get results above threshold
        caesar_results = try_caesar_shifts(untransposed, breaker, fitness_threshold)
        
        # Add grid info to each result
        for result in caesar_results:
            result["first_grid"] = first_grid
            result["second_grid"] = second_grid
            all_results.append(result)
    
    return all_results

def break_caesar_transposition_cipher(ciphertext, quadgram_file, double=False, fitness_threshold=90, processes=None):
    """Break a combined columnar transposition (single or double) + Caesar substitution cipher."""
    if not processes:
        processes = max(1, multiprocessing.cpu_count() - 1)  # Leave one CPU free
    
    # Get grid sizes
    text_length = len(ciphertext)
    grid_sizes = find_grid_sizes(text_length)
    
    if not grid_sizes:
        print("No valid grid sizes found for this text length.")
        return []
    
    print(f"Found {len(grid_sizes)} possible grid sizes")
    
    start_time = time.time()
    all_results = []
    
    if double:
        # For double transposition, we need to try all combinations of grid sizes
        print(f"Using DOUBLE transposition with Caesar cipher")
        
        # Generate all possible combinations of two grid sizes
        grid_combinations = []
        for first_grid in grid_sizes:
            for second_grid in grid_sizes:
                grid_combinations.append((first_grid, second_grid))
        
        print(f"Testing {len(grid_combinations)} grid combinations with 26 Caesar shifts each")
        print(f"Total combinations: {len(grid_combinations) * 26}")
        
        # Divide work into batches
        batch_size = max(1, len(grid_combinations) // processes)
        batches = []
        
        for i in range(0, len(grid_combinations), batch_size):
            end_idx = min(i + batch_size, len(grid_combinations))
            batch_combinations = grid_combinations[i:end_idx]
            batches.append((ciphertext, quadgram_file, batch_combinations, fitness_threshold))
        
        # Process batches
        with ProcessPoolExecutor(max_workers=processes) as executor:
            for batch_results in executor.map(break_caesar_double_transposition, batches):
                all_results.extend(batch_results)
    
    else:
        # For single transposition
        print(f"Using SINGLE transposition with Caesar cipher")
        print(f"Testing {len(grid_sizes)} grid sizes with 26 Caesar shifts each")
        print(f"Total combinations: {len(grid_sizes) * 26}")
        
        # Divide work into batches
        batch_size = max(1, len(grid_sizes) // processes)
        batches = []
        
        for i in range(0, len(grid_sizes), batch_size):
            end_idx = min(i + batch_size, len(grid_sizes))
            batch_grid_sizes = grid_sizes[i:end_idx]
            batches.append((ciphertext, quadgram_file, batch_grid_sizes, fitness_threshold))
        
        # Process batches
        with ProcessPoolExecutor(max_workers=processes) as executor:
            for batch_results in executor.map(break_caesar_single_transposition, batches):
                all_results.extend(batch_results)
    
    total_time = time.time() - start_time
    print(f"\nTotal processing time: {total_time:.2f}s")
    
    # Sort results by fitness
    all_results.sort(key=lambda x: x["fitness"], reverse=True)
    
    print(f"Found {len(all_results)} results with fitness above {fitness_threshold}")
    
    return all_results

def break_single_substitution(task_args):
    """Break the substitution cipher on a single transposed text"""
    untransposed, quadgram_file, rows, cols, max_rounds, consolidate = task_args
    
    # Initialize the breaker
    start_time = time.time()
    with open(quadgram_file, 'r') as f:
        breaker = subbreaker.Breaker(f)
    
    # Break the substitution cipher
    result = breaker.break_cipher(
        untransposed,
        max_rounds=max_rounds,
        consolidate=consolidate
    )
    
    elapsed_time = time.time() - start_time
    
    # Create a Key object from the result key string
    key_obj = subbreaker.Key(result.key, alphabet=result.alphabet)
    
    # Decode the text using the key object
    decrypted = key_obj.decode(untransposed)
    
    return {
        "rows": rows,
        "cols": cols,
        "untransposed": untransposed,
        "decrypted": decrypted,
        "key": result.key,  # String representation of the key
        "fitness": result.fitness,
        "keys_tried": result.nbr_keys,
        "elapsed_time": elapsed_time
    }

def main():
    parser = argparse.ArgumentParser(description="Break a columnar transposition with substitution cipher")
    parser.add_argument("--text", type=str, help="The ciphertext to break")
    parser.add_argument("--file", type=str, help="File containing the ciphertext")
    parser.add_argument("--quadgrams", type=str, default="EN.json", help="Path to the quadgrams file")
    parser.add_argument("--max-rounds", type=int, default=10000, help="Maximum hill climbing rounds")
    parser.add_argument("--consolidate", type=int, default=3, help="Consolidation parameter")
    parser.add_argument("--processes", type=int, default=8, help="Number of processes to use")
    parser.add_argument("--output", type=str, help="Output file for results")
    parser.add_argument("--double", action="store_true", help="Use double transposition (default is single)")
    parser.add_argument("--caesar", action="store_true", help="Use Caesar cipher brute force instead of hill climbing")
    parser.add_argument("--threshold", type=float, default=90, help="Fitness threshold for Caesar results")
    
    args = parser.parse_args()
    
    # Get ciphertext
    if args.text:
        ciphertext = args.text
    elif args.file:
        with open(args.file, 'r') as f:
            ciphertext = f.read().strip()
    else:
        ciphertext = input("Enter the ciphertext: ").strip()
    
    # Clean the ciphertext - keep only alphabetic characters
    clean_ciphertext = ''.join(c.lower() for c in ciphertext if c.isalpha())
    
    print(f"Breaking cipher text (length {len(clean_ciphertext)})...")
    
    # Choose the algorithm based on arguments
    if args.caesar:
        results = break_caesar_transposition_cipher(
            clean_ciphertext,
            args.quadgrams,
            double=args.double,
            fitness_threshold=args.threshold,
            processes=args.processes
        )
    elif args.double:
        print("Using DOUBLE columnar transposition mode with hill climbing")
        results = break_double_transposition_cipher(
            clean_ciphertext, 
            args.quadgrams,
            max_rounds=args.max_rounds,
            consolidate=args.consolidate,
            processes=args.processes
        )
    else:
        print("Using SINGLE columnar transposition mode with hill climbing")
        results = break_combined_cipher(
            clean_ciphertext, 
            args.quadgrams,
            max_rounds=args.max_rounds,
            consolidate=args.consolidate,
            processes=args.processes
        )
    
    # Display and save results
    if results:
        print("\nTop results (sorted by fitness):")
        
        for i, result in enumerate(results[:5]):  # Show top 5
            if "shift" in result:  # Caesar result
                if "first_grid" in result:  # Double transposition
                    first_grid = result["first_grid"]
                    second_grid = result["second_grid"]
                    print(f"\n{i+1}. First grid: {first_grid[0]}×{first_grid[1]}, "
                          f"Second grid: {second_grid[0]}×{second_grid[1]}, "
                          f"Caesar shift: {result['shift']}")
                else:  # Single transposition
                    print(f"\n{i+1}. Grid size: {result['rows']} rows × {result['cols']} columns, "
                          f"Caesar shift: {result['shift']}")
            elif "first_grid" in result:  # Regular double transposition
                first_grid = result["first_grid"]
                second_grid = result["second_grid"]
                print(f"\n{i+1}. First grid: {first_grid[0]}×{first_grid[1]}, "
                      f"Second grid: {second_grid[0]}×{second_grid[1]}")
            else:  # Regular single transposition
                print(f"\n{i+1}. Grid size: {result['rows']} rows × {result['cols']} columns")
                
            print(f"   Fitness: {result['fitness']:.2f}")
            if "key" in result:
                print(f"   Key: {result['key']}")
            print(f"   Decrypted: {result['decrypted'][:100]}..." if len(result['decrypted']) > 100 else result['decrypted'])
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nFull results saved to {args.output}")
    else:
        print("No results found.")

# Keep the original single transposition function for backward compatibility
def break_combined_cipher(ciphertext, quadgram_file, max_rounds=10000, consolidate=3, processes=None):
    """Break a combined columnar transposition and substitution cipher"""
    if not processes:
        processes = max(1, multiprocessing.cpu_count() - 1)  # Leave one CPU free
    
    # Get all possible grid sizes
    grid_sizes = find_grid_sizes(len(ciphertext))
    
    if not grid_sizes:
        print("No valid grid sizes found for this text length.")
        return []
    
    print(f"Found {len(grid_sizes)} possible grid sizes:")
    for i, (rows, cols) in enumerate(grid_sizes):
        print(f"{i+1}. {rows} rows × {cols} columns")
    
    # Prepare tasks for parallel processing
    tasks = []
    
    for rows, cols in grid_sizes:
        # Reverse the columnar transposition
        untransposed = reverse_columnar_transposition(ciphertext, rows, cols)
        
        # Create task for breaking substitution
        tasks.append((untransposed, quadgram_file, rows, cols, max_rounds, consolidate))
    
    # Process tasks in parallel
    results = []
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=processes) as executor:
        # Use break_single_substitution instead of break_substitution
        for i, result in enumerate(executor.map(break_single_substitution, tasks)):
            results.append(result)
            
            rows, cols = result["rows"], result["cols"]
            print(f"\nCompleted {i+1}/{len(tasks)}: {rows}×{cols} grid")
            print(f"Fitness: {result['fitness']:.2f}")
            print(f"Keys tried: {result['keys_tried']}")
            print(f"Time: {result['elapsed_time']:.2f}s")
    
    total_time = time.time() - start_time
    print(f"\nTotal processing time: {total_time:.2f}s")
    
    # Sort results by fitness
    results.sort(key=lambda x: x["fitness"], reverse=True)
    
    return results

if __name__ == "__main__":
    # For Windows compatibility
    multiprocessing.freeze_support()
    main()