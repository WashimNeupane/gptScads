import argparse
import json
import os

def load_config_from_json(config_dir, json_filename):
    """Load configuration from a JSON file."""
    config_path = os.path.join(config_dir, json_filename + '.json')
    if not os.path.exists(config_path):
        print(f"Error: JSON file '{json_filename}' not found in the '{config_dir}' directory.")
        exit(1)

    with open(config_path, 'r') as f:
        return json.load(f)

def parse_configs():
    # Define command-line arguments parser
    parser = argparse.ArgumentParser(description='Process configurations from JSON files')

    # Define command-line arguments for method configuration
    parser.add_argument('--method_config', type=str, default='method.json', help='Path to method configuration JSON file (without extension)')
    parser.add_argument('-m', '--method', type=str, nargs='+', help='Activate specific method(s) from the method configuration file')

    # Define command-line arguments for PEFT configuration
    parser.add_argument('-c', '--peft_config', type=str, default=None, help='Path to PEFT configuration JSON file (without extension)')

    # Define command-line arguments for training configuration
    parser.add_argument('-t', '--training_config', type=str, default='default', help='Path to training configuration JSON file (without extension)')

    args = parser.parse_args()

    # Check if '-m use_peft' and '-c' are either both specified or both absent
    if args.method is not None and 'use_peft' in args.method:
        if args.peft_config is None:
            raise ValueError("'-c' argument must be specified with '-m use_peft'.")
    elif args.peft_config is not None:
        raise ValueError("'-m use_peft' argument must be specified with '-c'.")

    # Load method configuration from JSON file
    method_config_path = os.path.join('config', 'method', args.method_config)
    if not os.path.exists(method_config_path):
        raise FileNotFoundError(f"Method configuration file '{args.method_config}' not found.")
    with open(method_config_path, 'r') as f:
        method_config = json.load(f)

    # Activate specified methods
    if args.method:
        for method in args.method:
            if method not in method_config:
                raise ValueError(f"Method '{method}' is not defined in the method configuration file.")
            method_config[method] = True

    # Ensure only one method is active
    active_methods = [method for method, value in method_config.items() if value]
    if len(active_methods) > 1:
        raise ValueError(f"More than one method is active: {active_methods}. Only one method can be active at a time.")

    # Parse PEFT configuration if 'use_peft' is set to True
    if method_config.get("use_peft", False):
        # Load PEFT configuration from JSON file
        peft_config_dir = os.path.join('config', 'peft')
        peft_config = load_config_from_json(peft_config_dir, args.peft_config)
    else:
        peft_config = None

    # Load training configuration from JSON file
    training_config_dir = os.path.join('config', 'trainer')
    training_config = load_config_from_json(training_config_dir, args.training_config)

    return peft_config, training_config, method_config

