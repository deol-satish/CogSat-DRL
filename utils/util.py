import builtins
from utils.settings import debug_output

# Save the original print function
original_print = builtins.print

# Override the built-in print
def custom_print(*args, **kwargs):
    if debug_output:
        original_print(*args, **kwargs)

# Apply the override
builtins.print = custom_print
