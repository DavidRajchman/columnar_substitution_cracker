# Columnar Transposition + Substitution Cipher Breaker

This documentation provides guidance on using the columnar_subreaker.py tool, a powerful application for breaking combined columnar transposition and substitution ciphers.

## Overview

This tool can break:
- Single columnar transposition + general substitution
- Double columnar transposition + general substitution
- Single columnar transposition + Caesar cipher
- Double columnar transposition + Caesar cipher

It uses statistical language analysis based on quadgrams to evaluate potential solutions and leverages parallel processing to efficiently search the solution space.

## Dependencies

Required Python libraries:
- `subbreaker` - The core substitution cipher breaking library
- `concurrent.futures` - For parallel processing
- Standard libraries: `os`, `time`, `json`, `argparse`, `multiprocessing`

## Installation

1. Install Python 3.6+ (or PyPy for better performance)

2. Install the required dependencies:
   ```bash
   pip install subbreaker
   ```

3. Download the quadgram file (e.g., EN.json) for your target language. This contains language statistics used to evaluate decryption attempts.

## Using PyPy for Enhanced Performance

This tool runs significantly faster for longer workloads with PyPy, especially for large ciphers or double transposition:

1. Install PyPy from https://www.pypy.org/download.html

2. Install dependencies with PyPy:
   ```bash
   pypy -m pip install subbreaker
   ```

3. Run the tool with PyPy:
   ```bash
   pypy columnar_subreaker.py [options]
   ```

## Basic Usage

```bash
python columnar_subreaker.py --file encrypted.txt --quadgrams EN.json
```

## Command-Line Arguments

| Argument | Description |
|----------|-------------|
| `--text TEXT` | Input ciphertext directly as a string |
| `--file FILE` | Read ciphertext from a file |
| `--quadgrams FILE` | Path to the quadgram JSON file (default: EN.json) |
| `--double` | Use double columnar transposition (default: single) |
| `--caesar` | Use Caesar cipher brute force instead of general substitution |
| `--threshold FLOAT` | Minimum fitness threshold for Caesar cipher results (default: 90) |
| `--max-rounds INT` | Maximum hill climbing rounds for substitution breaking (default: 10000) |
| `--consolidate INT` | Number of times the best local maximum must be found (default: 3) |
| `--processes INT` | Number of parallel processes to use (default: CPU number of threads -1) |
| `--output FILE` | Save full results to a JSON file |

## Breaking Different Cipher Types

### General Substitution with Single Transposition
```bash
python columnar_subreaker.py --file encrypted.txt --quadgrams EN.json
```

### General Substitution with Double Transposition
```bash
python columnar_subreaker.py --file encrypted.txt --quadgrams EN.json --double
```

### Caesar Cipher with Single Transposition
```bash
python columnar_subreaker.py --file encrypted.txt --quadgrams EN.json --caesar
```

### Caesar Cipher with Double Transposition
```bash
python columnar_subreaker.py --file encrypted.txt --quadgrams EN.json --caesar --double
```

## Tips for Short Texts

When working with short ciphers:

1. Use the Caesar cipher mode with a lower threshold:
   ```bash
   python columnar_subreaker.py --file short.txt --quadgrams EN.json --caesar --threshold 70
   ```

2. Try both single and double transposition modes to find the best fit.

3. Use PyPy for faster execution, which allows testing more combinations quickly.

## Output Interpretation

Results are sorted by fitness score, with higher scores indicating more likely solutions:
- A score of 100+ typically indicates correct English text
- Scores between 80-100 may still contain readable text with some errors
- Scores below 70 generally indicate incorrect solutions

For Caesar cipher results, the shift value is also displayed.

## Saving Results

Save full results to a JSON file for further analysis:

```bash
python columnar_subreaker.py --file encrypted.txt --quadgrams EN.json --output results.json
```

## Advanced Usage

### Performance Tuning

- Adjust the number of processes based on your CPU:
  ```bash
  python columnar_subreaker.py --file encrypted.txt --quadgrams EN.json --processes 4
  ```

- For large texts or many grid combinations, increase `--max-rounds` to find better solutions:
  ```bash
  python columnar_subreaker.py --file large.txt --quadgrams EN.json --max-rounds 20000
  ```

### Using with Very Large Files

For extremely large files, consider breaking only a portion:

1. Extract a representative sample (1000+ characters)
2. Break the sample to identify the correct grid dimensions and key
3. Apply the discovered key to the entire file

## Troubleshooting

- If no valid grid sizes are found, your text length may not be factorizable into dimensions ≥3
- If fitness scores are consistently low, try:
  - Using a different quadgram file for the appropriate language
  - Checking if the text actually uses a different cipher type
  - Lowering the threshold for Caesar cipher mode

## Example Workflow

1. First try Caesar cipher mode (faster):
   ```bash
   python columnar_subreaker.py --file cipher.txt --quadgrams EN.json --caesar --threshold 80
   ```

2. If no good results, try general substitution:
   ```bash
   python columnar_subreaker.py --file cipher.txt --quadgrams EN.json
   ```

3. If still unsuccessful, try double transposition:
   ```bash
   python columnar_subreaker.py --file cipher.txt --quadgrams EN.json --double
   ```

4. Save promising results:
   ```bash
   python columnar_subreaker.py --file cipher.txt --quadgrams EN.json --caesar --output results.json
   ```