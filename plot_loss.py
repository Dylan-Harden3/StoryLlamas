import re
import matplotlib.pyplot as plt

def plot_training_and_validation_losses(log_files):
    plt.figure(figsize=(12, 7))
    
    for file in log_files:
        steps = []
        train_losses = []
        val_losses = []
        val_steps = []

        train_log_pattern = re.compile(r"step\s+(\d+)/\d+\s+\|\s+loss:\s+([\d.]+)")
        val_log_pattern = re.compile(r"val loss:\s+([\d.]+)")

        with open(file, "r") as f:
            for line in f:
                train_match = train_log_pattern.search(line)
                if train_match:
                    steps.append(int(train_match.group(1)))
                    train_losses.append(float(train_match.group(2)))

                val_match = val_log_pattern.search(line)
                if val_match:
                    val_losses.append(float(val_match.group(1)))
                    val_steps.append(steps[-1] if steps else len(val_losses))
 
        plt.plot(steps, train_losses, label=f"{file} Training Loss", alpha=0.7)
        plt.plot(val_steps, val_losses, label=f"{file} Validation Loss", alpha=0.7)

    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss over Steps")
    plt.ylim(1.0, 5.0)
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

log_files = ["output_lr1_25k.txt", "output_lr3_25k.txt", "output_lr5_25k.txt", "output_lr3_25k_do.txt"]
plot_training_and_validation_losses(log_files)