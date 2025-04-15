import random
from datetime import datetime

LOG_FILE = "random_indices_log.txt"

def generate_random_indices(imax, rows, cols):
  """
  Generates a list of random integers similar to MATLAB's randi([1, imax], rows, cols),
  and logs the result to a file.
  Currently only supports generating a 1D array (rows=1).
  """
  if rows != 1:
    raise ValueError("This function currently only supports generating 1 row of random numbers.")
  
  # Ensure imax is treated as an integer
  imax_int = int(imax)
  num_cols_int = int(cols)

  # Generate 'cols' number of random integers between 1 and imax_int (inclusive)
  indices = [random.randint(1, imax_int) for _ in range(num_cols_int)]

  # Log the generated indices
  try:
    with open(LOG_FILE, 'a') as f:
      timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
      f.write(f"{timestamp} - Generated {num_cols_int} indices (max={imax_int}): {indices}\n")
  except Exception as e:
    print(f"Warning: Could not write to log file {LOG_FILE}. Error: {e}")

  return indices

# Example usage (can be run directly in Python)
if __name__ == "__main__":
  imax_val = 10
  num_cols = 5
  random_list = generate_random_indices(imax_val, 1, num_cols)
  print(f"Generated {num_cols} random indices between 1 and {imax_val}: {random_list}")
  print(f"Check '{LOG_FILE}' for the logged output.")