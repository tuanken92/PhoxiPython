import time
from datetime import datetime
import json

def current_milli_time():
    return round(time.time() * 1000)

def print_time():
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)

# Define a function to read the configuration from the JSON file
def read_config(filename):
    try:
        with open(filename, 'r') as file:
            config = json.load(file)
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file '{filename}' not found.")
        return {}