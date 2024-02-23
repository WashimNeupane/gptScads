import json
import math
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM
from datasets import load_from_disk
from distillation import init_student_model_from_teacher
from parser import parse_configs
from peft import get_peft_model, get_peft_config

def main():
    # Parse configurations
    peft_config, training_config, method_config = parse_configs()
    print("PEFT Configuration:")
    print(json.dumps(peft_config, indent=4))
    print("Training Configuration:")
    print(json.dumps(training_config, indent=4))
    print("Method Configuration:")
    print(json.dumps(method_config, indent=4))

    # Default config directory
    config_dir = 'config'

    # Load the data
    lm_datasets_train = load_from_disk("data/processed_train_dataset")
    lm_datasets_val = load_from_disk("data/processed_validation_dataset")

    # Load the GPT2 model with LM head for causal language modeling
    base_model = "gpt2"
    model = AutoModelForCausalLM.from_pretrained(base_model)

    if method_config.get("use_student", False):
        init_student_model_from_teacher(base_model)
        print("Student model initialized.")
        model.save_pretrained("model_student")
        print("Student model saved.")
    else:
        if method_config.get("use_peft", False):
            peft_config.pop("base_model")
            peft_config = get_peft_config(peft_config)  # Remove base model path from PEFT config
            model = get_peft_model(model, peft_config)
            print("Peft model training instantiated")
            model.print_trainable_parameters()
        run_training(model, training_config, lm_datasets_train, lm_datasets_val, method_config)

        # Save the trained model
        model.save_pretrained("model")
        print("Model saved.")

def run_training(model, training_config, train_dataset, val_dataset, method_config):
    if method_config.get("use_student", False):
        return  # No training loop needed for student model initialization
    else:
        # Create TrainingArguments object from loaded dictionary
        try:
            training_args = TrainingArguments(**training_config)
        except ValueError as e:
            print("Error creating TrainingArguments:", e)
            print("Mixed precision training is not supported on this device. Disabling mixed precision training.")
            training_config['fp16'] = False
            training_args = TrainingArguments(**training_config)

        # Define trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )

        # Train the model
        print("Starting training...")
        trainer.train()

        # Evaluate the model
        eval_results = trainer.evaluate()
        print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")


if __name__ == "__main__":
    main()
