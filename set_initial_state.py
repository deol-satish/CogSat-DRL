import json
from datetime import datetime

def set_initial_state(state_dict):
    """Save satellite state dictionary to a text file with timestamp"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"satellite_state_{timestamp}.txt"
        
        with open(filename, 'w') as f:
            f.write(f"Satellite State at {timestamp}\n")
            f.write("="*40 + "\n")
            f.write(json.dumps(state_dict, indent=4))
        
        print(f"Successfully saved state to {filename}")
        return True  # Explicitly return True on success
        
    except Exception as e:
        print(f"Error saving state: {str(e)}")
        return False  # Explicitly return False on failure