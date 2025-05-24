import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import snntorch as snn
from snntorch import surrogate
import psutil
import gc
import subprocess
import platform
import threading
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, classification_report
import tonic
import tonic.transforms as transforms
from tqdm import tqdm

# ================= CONFIGURATION =================
# Define paths
dataset_root = os.path.expanduser("./DVSGesturedataset")
cache_dir = os.path.join(dataset_root, "cached_tensors")

# Hyperparameters
num_epochs = 3
batch_size = 16
learning_rate = 0.001
time_steps = 16  # Number of time steps for each sample
num_classes = 11  # 11 gesture types (0-10)

# Gesture class names (for visualizations)
gesture_class_names = [
    "Hand Clapping",
    "Right Hand Wave", 
    "Left Hand Wave", 
    "Right Arm Clockwise", 
    "Right Arm Counter Clockwise", 
    "Left Arm Clockwise", 
    "Left Arm Counter Clockwise", 
    "Arm Roll", 
    "Air Drums", 
    "Air Guitar", 
    "Other"
]

# ================= CACHED DATASET CLASS =================
class CachedDVSGestureDataset(Dataset):
    """
    A dataset wrapper that processes DVSGesture data once and caches it to disk.
    This dramatically improves training speed by avoiding repeated on-the-fly processing.
    """
    
    def __init__(self, 
                root_dir=dataset_root,
                cache_dir=cache_dir,
                train=True,
                time_steps=time_steps,
                force_reprocess=False):
        """
        Args:
            root_dir: Directory where the original DVSGesture dataset is/will be stored
            cache_dir: Directory to store cached tensors
            train: Whether to use the training or testing set
            time_steps: Number of time bins to use when discretizing events
            force_reprocess: If True, reprocess data even if cache exists
        """
        self.root_dir = os.path.expanduser(root_dir)
        self.cache_dir = os.path.expanduser(cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.train = train
        self.time_steps = time_steps
        self.split_name = "train" if train else "test"
        
        # Set up file paths for cached data
        self.frames_file = os.path.join(self.cache_dir, f"{self.split_name}_frames.pt")
        self.labels_file = os.path.join(self.cache_dir, f"{self.split_name}_labels.pt")
        
        # Check if cached data exists
        if os.path.exists(self.frames_file) and os.path.exists(self.labels_file) and not force_reprocess:
            print(f"Loading cached {self.split_name} data from {self.cache_dir}")
            self.frames = torch.load(self.frames_file)
            self.labels = torch.load(self.labels_file)
        else:
            print(f"Processing {self.split_name} data and caching to {self.cache_dir}")
            self._process_and_cache_data()
            
        # Print dataset information
        print(f"Dataset ready: {len(self)} {self.split_name} samples, frame shape: {self.frames[0].shape}")
    
    def _process_and_cache_data(self):
        """Process the Tonic dataset and cache to disk."""
        # Define sensor size for DVSGesture (128x128)
        sensor_size = tonic.datasets.DVSGesture.sensor_size
        
        # Define transforms for event data
        frame_transform = transforms.Compose([
            # Denoise the event stream
            transforms.Denoise(filter_time=10000),
            # Convert events to frames (2-channel: ON/OFF events)
            transforms.ToFrame(
                sensor_size=sensor_size,
                n_time_bins=self.time_steps
            )
        ])
        
        # Create the Tonic dataset
        tonic_dataset = tonic.datasets.DVSGesture(
            save_to=self.root_dir,
            transform=frame_transform,
            train=self.train
        )
        
        # Process all samples
        all_frames = []
        all_labels = []
        
        print(f"Processing {len(tonic_dataset)} samples from {self.split_name} set...")
        
        # Create a tqdm progress bar
        for i, (frame, label) in enumerate(tqdm(tonic_dataset, desc=f"Processing {self.split_name} set")):
            # Ensure frame is a tensor
            if not isinstance(frame, torch.Tensor):
                frame = torch.tensor(frame, dtype=torch.float32)
            elif frame.dtype != torch.float32:
                frame = frame.float()
            
            # For SNNs, we need data in format [timesteps, channels, height, width]
            # But Tonic typically returns [channels, timesteps, height, width]
            if frame.dim() == 4 and frame.shape[0] == 2:
                # Format is [channels, timesteps, height, width]
                frame = frame.permute(1, 0, 2, 3)
            
            all_frames.append(frame)
            all_labels.append(label)
        
        # Stack all frames and labels
        self.frames = torch.stack(all_frames)
        self.labels = torch.tensor(all_labels, dtype=torch.long)
        
        # Save to disk
        torch.save(self.frames, self.frames_file)
        torch.save(self.labels, self.labels_file)
        print(f"Cached {len(self.frames)} samples to {self.cache_dir}")
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.frames[idx], self.labels[idx]

# ================= POWER MONITOR CLASS =================
class PowerMonitor:
    def __init__(self, sample_interval=0.5):
        """
        Initialize the power monitor.
        
        Args:
            sample_interval: Time between samples in seconds
        """
        self.sample_interval = sample_interval
        self.power_readings = []
        self.timestamps = []
        self.monitoring = False
        self.monitor_thread = None
        self.is_macos = platform.system() == "Darwin"
        self.is_nvidia = torch.cuda.is_available()
        
    def start_monitoring(self):
        """Start the power monitoring thread."""
        # Check if we can monitor power on this system
        if not self.is_macos and not self.is_nvidia:
            print("Power monitoring is only available on macOS with MPS or systems with NVIDIA GPUs.")
            return
            
        self.monitoring = True
        self.power_readings = []
        self.timestamps = []
        self.monitor_thread = threading.Thread(target=self._monitor_power)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        print("Power monitoring started.")
        
    def stop_monitoring(self):
        """Stop the power monitoring thread and return summary statistics."""
        if not self.monitoring:
            return None
            
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
            
        if not self.power_readings:
            return None
            
        # Calculate statistics
        power_array = np.array(self.power_readings)
        stats = {
            "min": np.min(power_array),
            "max": np.max(power_array),
            "mean": np.mean(power_array),
            "median": np.median(power_array),
            "std": np.std(power_array),
            "samples": len(power_array),
            "total_energy": self._calculate_energy()
        }
        
        print(f"Power monitoring stopped. Collected {stats['samples']} samples.")
        return stats
        
    def _monitor_power(self):
        """Background thread that continuously samples power consumption."""
        while self.monitoring:
            try:
                gpu_power = None
                
                if self.is_macos:
                    # macOS MPS power monitoring using powermetrics
                    cmd = ["sudo", "powermetrics", "--samplers", "gpu_power", "-n", "1", "-i", "100"]
                    power_output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=15).decode("utf-8")
                    
                    # Extract GPU power value
                    power_lines = [line for line in power_output.split("\n") if "GPU Power" in line]
                    if power_lines:
                        power_str = power_lines[0].split(":")[-1].strip()
                        gpu_power_mw = float(power_str.split()[0])  # Extract numerical value
                        
                        # Convert from milliwatts to watts
                        gpu_power = gpu_power_mw / 1000.0
                
                elif self.is_nvidia:
                    # NVIDIA GPU power monitoring using nvidia-smi
                    cmd = ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"]
                    power_output = subprocess.check_output(cmd).decode("utf-8").strip()
                    
                    try:
                        # Convert string to float (power in watts)
                        gpu_power = float(power_output)
                    except ValueError:
                        print(f"Could not parse power value: {power_output}")
                
                if gpu_power is not None:
                    self.power_readings.append(gpu_power)
                    self.timestamps.append(time.time())
                    
            except Exception as e:
                print(f"Error in power monitoring: {e}")
                
            # Sleep for the sample interval
            time.sleep(self.sample_interval)
            
    def _calculate_energy(self):
        """
        Calculate total energy consumption in joules (watt-seconds)
        using the trapezoidal rule for integration
        """
        if len(self.timestamps) < 2:
            return 0
            
        # Calculate time differences and convert to seconds
        times = np.array(self.timestamps)
        time_diffs = np.diff(times)
        
        # Power values in watts
        powers = np.array(self.power_readings)
        
        # Calculate energy using trapezoidal rule: E = ∫P(t)dt
        # For each time segment, average the power at the beginning and end
        energy_segments = time_diffs * (powers[:-1] + powers[1:]) / 2
        
        # Sum all energy segments to get total energy in joules (watt-seconds)
        total_energy = np.sum(energy_segments)
        
        return total_energy
        
    def get_power_profile(self):
        """Get the power profile for plotting."""
        if not self.timestamps or not self.power_readings:
            return None
            
        # Convert timestamps to relative time in seconds from the start
        start_time = self.timestamps[0]
        relative_times = [t - start_time for t in self.timestamps]
        
        return relative_times, self.power_readings

# ================= DEVICE CONFIGURATION =================
# Device configuration
print("PyTorch version:", torch.__version__)
print("MPS available:", torch.backends.mps.is_available())
print("MPS built:", torch.backends.mps.is_built())

if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
    print("Using MPS device")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA device")
else:
    device = torch.device("cpu")
    print("Using CPU device")

# ================= SNN MODEL =================
# Define SNN architecture with snnTorch
class DVSSNN(nn.Module):
    def __init__(self, num_classes=11, beta=0.5):
        super(DVSSNN, self).__init__()
        
        # Define surrogate gradient function
        spike_grad = surrogate.atan(alpha=2.0)
        
        # layers
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.pool2 = nn.MaxPool2d(2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.pool3 = nn.MaxPool2d(2)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.lif4 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.pool4 = nn.MaxPool2d(2)
        
        # Calculate the size after multiple pooling operations (128x128 -> 16x16)
        self.fc_size = 256 * 16 * 16
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.fc_size, 512)
        self.lif5 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
        self.fc2 = nn.Linear(512, num_classes)
        self.lif_out = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
    def forward(self, x):
        # x shape: [batch, timesteps, channels, height, width]
        batch_size = x.size(0)
        num_steps = x.size(1)
        
        # Initialize neuron states
        mem1 = torch.zeros(batch_size, 32, 128, 128, device=device)
        mem2 = torch.zeros(batch_size, 64, 128, 128, device=device)
        mem3 = torch.zeros(batch_size, 128, 64, 64, device=device)
        mem4 = torch.zeros(batch_size, 256, 32, 32, device=device)
        mem5 = torch.zeros(batch_size, 512, device=device)
        mem_out = torch.zeros(batch_size, num_classes, device=device)
        
        # Spike accumulator for output
        spk_out_acc = torch.zeros(batch_size, num_classes, device=device)
        
        # Temporal processing
        for t in range(num_steps):
            # Get current timestep input
            x_t = x[:, t]
            
            # Conv layer 1
            cur1 = self.bn1(self.conv1(x_t))
            spk1, mem1 = self.lif1(cur1, mem1)
            
            # Conv layer 2
            cur2 = self.bn2(self.conv2(spk1))
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2 = self.pool2(spk2)
            
            # Conv layer 3
            cur3 = self.bn3(self.conv3(spk2))
            spk3, mem3 = self.lif3(cur3, mem3)
            spk3 = self.pool3(spk3)
            
            # Conv layer 4
            cur4 = self.bn4(self.conv4(spk3))
            spk4, mem4 = self.lif4(cur4, mem4)
            spk4 = self.pool4(spk4)
            
            # Flatten
            flat = spk4.view(batch_size, -1)
            
            # FC layer 1
            cur5 = self.fc1(flat)
            spk5, mem5 = self.lif5(cur5, mem5)
            
            # FC layer 2 (output)
            cur_out = self.fc2(spk5)
            spk_out, mem_out = self.lif_out(cur_out, mem_out)
            
            # Accumulate output spikes
            spk_out_acc = spk_out_acc + spk_out
        
        # Return rate-coded output (accumulated spikes)
        return spk_out_acc

# ================= NEW UTILITY FUNCTIONS =================
def sync_device():
    """Ensure operations are completed on device"""
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()

def log_memory_usage(stage=""):
    """Logs RAM and GPU memory usage for a given training stage and returns a formatted string."""
    # Ensure device is synchronized before measuring
    sync_device()
    
    # RAM usage
    ram_usage = psutil.virtual_memory().used / 1e9  # Convert to GB
    memory_log = f"[{stage}] RAM Usage: {ram_usage:.2f} GB"
    
    # Device-specific metrics
    if device.type == "mps":
        try:
            allocated = torch.mps.current_allocated_memory() / 1e9  # Convert to GB
            driver_mem = torch.mps.driver_allocated_memory() / 1e9  # Convert to GB
            memory_log += f", GPU Allocated Memory: {allocated:.2f} GB, GPU Driver Memory: {driver_mem:.2f} GB"
        except Exception as e:
            memory_log += f", MPS memory info error: {e}"
    
    elif device.type == "cuda":
        allocated = torch.cuda.memory_allocated() / 1e9  # Convert to GB
        reserved = torch.cuda.memory_reserved() / 1e9  # Convert to GB
        memory_log += f", GPU Allocated Memory: {allocated:.2f} GB, GPU Reserved Memory: {reserved:.2f} GB"
        
        # Get GPU power usage (NVIDIA only)
        try:
            power_output = subprocess.check_output(["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader"], encoding="utf-8")
            memory_log += f", GPU Power Consumption: {power_output.strip()}"
        except Exception as e:
            pass
    
    # Return both a string for logging and a dictionary with resource usage
    resource_data = {
        'ram': ram_usage,
        'gpu_allocated': 0,
        'gpu_reserved': 0,
        'gpu_power': 0
    }
    
    if device.type == "mps":
        try:
            resource_data['gpu_allocated'] = allocated
            resource_data['gpu_reserved'] = driver_mem
        except:
            pass
    elif device.type == "cuda":
        resource_data['gpu_allocated'] = allocated
        resource_data['gpu_reserved'] = reserved
        try:
            # Extract numeric power value if possible
            power_val = float(power_output.strip().split()[0])
            resource_data['gpu_power'] = power_val
        except:
            pass
    
    print(memory_log)
    return resource_data

# ================= METRICS FUNCTIONS =================
def plot_confusion_matrix(y_true, y_pred, classes=None, epoch=None, save_dir='training_results_DVSGesture_SNN'):
    """
    Calculate and plot confusion matrix
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        classes: List of class names
        epoch: Epoch number (None for final evaluation)
        save_dir: Directory to save the plot
    """
    # Create directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Use numerical class indices if no class names provided
    if classes is None:
        classes = [str(i) for i in range(num_classes)]
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Convert to percentage (normalize by true labels)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create figure
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    
    if epoch is not None:
        plt.title(f'Confusion Matrix (Epoch {epoch+1})')
        filename = f'confusion_matrix_epoch_{epoch+1}.png'
    else:
        plt.title('Final Confusion Matrix')
        filename = 'final_confusion_matrix.png'
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()
    
    return cm

def calculate_metrics(y_true, y_pred, num_classes=11):
    """
    Calculate precision, recall, and F1 score
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        num_classes: Number of classes
        
    Returns:
        dict: Dictionary containing precision, recall, and F1 score for each class
    """
    # Calculate precision, recall, and F1 score for each class
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=range(num_classes), average=None
    )
    
    # Calculate macro and weighted averages
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro'
    )
    
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
    )
    
    # Create a dictionary with all metrics
    metrics = {
        'class_precision': precision,
        'class_recall': recall,
        'class_f1': f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1
    }
    
    return metrics

def calculate_power_metrics(results):
    """Calculate detailed power and energy metrics from training results."""
    if not results.get("epoch_power_stats"):
        return None
        
    # Initialize metrics dictionary
    metrics = {
        "peak_power": 0,
        "average_power": 0,
        "total_energy_joules": 0,
        "total_energy_wh": 0,
        "energy_per_epoch": [],
        "power_efficiency": [],  # Energy per accuracy improvement
    }
    
    # Extract and summarize metrics
    peak_powers = []
    mean_powers = []
    total_energy = 0
    
    for i, stats in enumerate(results["epoch_power_stats"]):
        peak_powers.append(stats["max"])
        mean_powers.append(stats["mean"])
        total_energy += stats["total_energy"]
        metrics["energy_per_epoch"].append(stats["total_energy"])
        
        # Calculate power efficiency (joules per percentage point of accuracy)
        if i > 0:
            accuracy_improvement = results["test_acc"][i] - results["test_acc"][i-1]
            if accuracy_improvement > 0:
                efficiency = stats["total_energy"] / accuracy_improvement
                metrics["power_efficiency"].append(efficiency)
    
    # Update metrics dictionary
    metrics["peak_power"] = max(peak_powers) if peak_powers else 0
    metrics["average_power"] = sum(mean_powers) / len(mean_powers) if mean_powers else 0
    metrics["total_energy_joules"] = total_energy
    metrics["total_energy_wh"] = total_energy / 3600  # Convert joules to watt-hours
    
    return metrics

# ================= VISUALIZATION FUNCTIONS =================
def plot_metrics_per_class(results, classes=None, results_dir="training_results_DVSGesture_SNN"):
    """Plot precision, recall, and F1 score for each class over epochs."""
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
    num_epochs = len(results["epoch_metrics"])
    if classes is None:
        classes = [str(i) for i in range(num_classes)]
    num_classes = len(classes)
    
    # Plot F1 score for each class over epochs
    plt.figure(figsize=(12, 8))
    for i in range(num_classes):
        class_f1 = [epoch_f1["class_f1"][i] for epoch_f1 in results["epoch_metrics"]]
        plt.plot(range(1, num_epochs+1), class_f1, 
                marker='o', label=classes[i])
    
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('F1 Score per Class Over Time')
    plt.grid(True)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(f"{results_dir}/f1_score_per_class.png")
    plt.close()
    
    # Plot overall macro metrics over epochs
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, num_epochs+1), [metric["macro_precision"] for metric in results["epoch_metrics"]], 
            'b-', marker='o', label='Macro Precision')
    plt.plot(range(1, num_epochs+1), [metric["macro_recall"] for metric in results["epoch_metrics"]], 
            'g-', marker='s', label='Macro Recall')
    plt.plot(range(1, num_epochs+1), [metric["macro_f1"] for metric in results["epoch_metrics"]], 
            'r-', marker='^', label='Macro F1')
    
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Overall Metrics Over Time')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{results_dir}/overall_metrics.png")
    plt.close()

def create_class_performance_dashboard(y_true, y_pred, classes=None, results=None, results_dir="training_results_DVSGesture_SNN"):
    """
    Create an integrated visualization showing confusion matrix alongside class metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        classes: List of class names
        results: Training results dictionary
        results_dir: Directory to save the visualization
    """
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
    if classes is None:
        classes = [str(i) for i in range(num_classes)]
    
    # Calculate metrics for final state
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Get final class metrics
    final_metrics = results["epoch_metrics"][-1]
    
    # Create a 2x2 subplot figure
    fig = plt.figure(figsize=(18, 16))
    
    # First subplot: Confusion Matrix
    ax1 = plt.subplot2grid((2, 2), (0, 0), rowspan=1, colspan=1)
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues', 
                xticklabels=classes, yticklabels=classes, ax=ax1)
    ax1.set_xlabel('Predicted labels')
    ax1.set_ylabel('True labels')
    ax1.set_title('Confusion Matrix (%)')
    
    # Second subplot: Class Precision, Recall, F1
    ax2 = plt.subplot2grid((2, 2), (0, 1), rowspan=1, colspan=1)
    
    # Prepare data for bar chart
    x = np.arange(len(classes))
    width = 0.25
    
    ax2.bar(x - width, final_metrics['class_precision'], width, label='Precision', color='blue', alpha=0.7)
    ax2.bar(x, final_metrics['class_recall'], width, label='Recall', color='green', alpha=0.7)
    ax2.bar(x + width, final_metrics['class_f1'], width, label='F1 Score', color='red', alpha=0.7)
    
    ax2.set_xlabel('Classes')
    ax2.set_ylabel('Score')
    ax2.set_title('Precision, Recall, and F1 Score per Class')
    ax2.set_xticks(x)
    ax2.set_xticklabels(classes, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Third subplot: Class accuracy bar chart
    ax3 = plt.subplot2grid((2, 2), (1, 0), rowspan=1, colspan=1)
    
    class_accuracy = np.diag(cm_percent)  # Diagonal elements of confusion matrix are per-class accuracy
    bars = ax3.bar(x, class_accuracy, color='purple', alpha=0.7)
    
    # Add value labels to bars
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom')
    
    ax3.set_xlabel('Classes')
    ax3.set_ylabel('Accuracy (%)')
    ax3.set_title('Per-class Accuracy')
    ax3.set_xticks(x)
    ax3.set_xticklabels(classes, rotation=45, ha='right')
    ax3.set_ylim(0, 110)  # Ensure space for percentage labels
    ax3.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Fourth subplot: Metrics over epochs for selected classes
    ax4 = plt.subplot2grid((2, 2), (1, 1), rowspan=1, colspan=1)
    
    # Get accuracy metrics over epochs from confusion matrices
    epochs = range(1, len(results["test_acc"]) + 1)
    
    # Select a few interesting classes (classes with highest/lowest accuracy)
    class_accuracies = np.diag(cm_percent)
    best_class_idx = np.argmax(class_accuracies)
    worst_class_idx = np.argmin(class_accuracies)
    
    # Plot F1 score over epochs for best and worst classes
    best_class_f1 = [epoch_metrics["class_f1"][best_class_idx] for epoch_metrics in results["epoch_metrics"]]
    worst_class_f1 = [epoch_metrics["class_f1"][worst_class_idx] for epoch_metrics in results["epoch_metrics"]]
    
    ax4.plot(epochs, best_class_f1, 'g-', marker='o', label=f'Best: {classes[best_class_idx]}')
    ax4.plot(epochs, worst_class_f1, 'r-', marker='s', label=f'Worst: {classes[worst_class_idx]}')
    ax4.plot(epochs, [metric["macro_f1"] for metric in results["epoch_metrics"]], 'b-', marker='^', label='Macro F1')
    
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('F1 Score')
    ax4.set_title('F1 Score Evolution')
    ax4.grid(True)
    ax4.legend()
    
    # Add an overall title
    plt.suptitle('DVS Gesture Recognition Performance (SNN): Class-Level Analysis', fontsize=16)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Adjust for main title
    plt.savefig(f"{results_dir}/class_performance_dashboard.png", dpi=300)
    plt.close()
    
    print(f"Class performance dashboard saved to {results_dir}/class_performance_dashboard.png")

def plot_training_results(results, classes=None):
    """Plot comprehensive training results including power profiles and metrics."""
    results_dir = "training_results_DVSGesture_SNN"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
    num_epochs = len(results["train_acc"])
    
    # Create figure with subplots
    fig, axs = plt.subplots(4, 1, figsize=(12, 20), 
                           gridspec_kw={'height_ratios': [1, 1, 1, 2]})
    
    # Plot 1: Accuracy
    axs[0].plot(range(1, num_epochs+1), results["train_acc"], 'b-', label='Train Accuracy')
    axs[0].plot(range(1, num_epochs+1), results["test_acc"], 'r-', label='Test Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Accuracy (%)')
    axs[0].set_title('Training and Testing Accuracy')
    axs[0].legend()
    axs[0].grid(True)
    
    # Plot 2: Epoch Times
    axs[1].plot(range(1, num_epochs+1), results["epoch_times"], 'g-')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Time (s)')
    axs[1].set_title('Epoch Completion Time')
    axs[1].grid(True)
    
    # Plot 3: Metrics over time
    axs[2].plot(range(1, num_epochs+1), [metric["macro_precision"] for metric in results["epoch_metrics"]], 'b-', label='Precision')
    axs[2].plot(range(1, num_epochs+1), [metric["macro_recall"] for metric in results["epoch_metrics"]], 'g-', label='Recall')
    axs[2].plot(range(1, num_epochs+1), [metric["macro_f1"] for metric in results["epoch_metrics"]], 'r-', label='F1 Score')
    axs[2].set_xlabel('Epoch')
    axs[2].set_ylabel('Score')
    axs[2].set_title('Overall Metrics Over Time')
    axs[2].legend()
    axs[2].grid(True)
    
    # Plot 4: Power Profile
    if "power_monitor" in results and results["power_monitor"] and results["power_monitor"].get_power_profile():
        times, powers = results["power_monitor"].get_power_profile()
        axs[3].plot(times, powers, 'r-')
        axs[3].set_xlabel('Time (s)')
        axs[3].set_ylabel('Power (W)')
        axs[3].set_title('GPU Power Consumption Profile')
        
        # Add horizontal lines for min, mean, max power
        if "epoch_power_stats" in results and results["epoch_power_stats"]:
            # Calculate overall statistics
            all_power = []
            for stats in results["epoch_power_stats"]:
                all_power.extend([stats["min"], stats["mean"], stats["max"]])
            
            min_power = min(all_power)
            max_power = max(all_power)
            mean_power = sum([s["mean"] for s in results["epoch_power_stats"]]) / len(results["epoch_power_stats"])
            
            axs[3].axhline(y=min_power, color='g', linestyle='--', label=f'Min: {min_power:.2f}W')
            axs[3].axhline(y=mean_power, color='b', linestyle='--', label=f'Mean: {mean_power:.2f}W')
            axs[3].axhline(y=max_power, color='r', linestyle='--', label=f'Max: {max_power:.2f}W')
            axs[3].legend()
        
        axs[3].grid(True)
    else:
        axs[3].text(0.5, 0.5, 'No power data available', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=axs[3].transAxes)
    
    plt.tight_layout()
    plt.savefig(f'{results_dir}/comprehensive_training_results.png')
    plt.close()
    
    # Also create a separate, more detailed power profile
    if "power_monitor" in results and results["power_monitor"] and results["power_monitor"].get_power_profile():
        times, powers = results["power_monitor"].get_power_profile()
        
        plt.figure(figsize=(14, 7))
        plt.plot(times, powers, 'r-', alpha=0.7)
        
        # Calculate moving average to show trend
        window_size = min(50, len(powers) // 10 or 1)  # Adaptive window size
        if len(powers) > window_size:
            moving_avg = np.convolve(powers, np.ones(window_size)/window_size, mode='valid')
            plt.plot(times[window_size-1:], moving_avg, 'b-', linewidth=2, label=f'{window_size}-point Moving Average')
        
        plt.xlabel('Time (s)')
        plt.ylabel('Power (W)')
        plt.title('Detailed GPU Power Consumption Profile')
        plt.grid(True)
        plt.legend()
        plt.savefig(f'{results_dir}/power_profile.png')
        plt.close()
    
    # If classes are available, create per-class metrics visualization
    if classes:
        plot_metrics_per_class(results, classes, results_dir)

# ================= TRAINING & EVALUATION FUNCTIONS =================
def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs):
    # Create results directory
    results_dir = "training_results_DVSGesture_SNN"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Initialize power monitor
    power_monitor = PowerMonitor(sample_interval=0.5)
    
    # Metrics storage
    metrics = {
        'train_acc': [],
        'test_acc': [],
        'epoch_times': [],
        'inference_times': [],
        'ram_usages': [],
        'gpu_allocated': [],
        'gpu_reserved': [],
        'epoch_power_stats': [],
        'epoch_metrics': [],
        'power_monitor': power_monitor
    }
    
    # Initial test
    print("Evaluating initial model performance...")
    initial_test_acc, initial_inference_time, all_targets, all_predictions = evaluate(model, test_loader, criterion, detailed=True)
    initial_metrics = calculate_metrics(all_targets, all_predictions, num_classes)
    print(f"Initial Test Accuracy: {initial_test_acc:.2f}%, Inference Time: {initial_inference_time:.4f}s, Macro F1: {initial_metrics['macro_f1']:.4f}")
    
    print(f"\nStarting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        # Start power monitoring
        power_monitor.start_monitoring()
        
        # Record start time
        start_time = time.time()
        
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(train_loader):
            # Move tensors to device
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Ensure previous operations complete
            sync_device()
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            loss.backward()
            optimizer.step()
            
            # Ensure optimization is complete
            sync_device()
            
            # Update statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Print batch progress (minimally)
            if (i + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
                # Periodically log GPU usage to confirm it's being utilized
                if (i + 1) % 50 == 0:
                    print("Current GPU usage during training:")
                    log_memory_usage("Training Batch")
        
        # Calculate epoch metrics
        epoch_time = time.time() - start_time
        train_accuracy = 100 * correct / total
        
        # Evaluate on test set
        test_accuracy, inference_time, all_targets, all_predictions = evaluate(model, test_loader, criterion)
        
        # Calculate advanced metrics
        epoch_metrics = calculate_metrics(all_targets, all_predictions, num_classes)
        
        # Get resource usage (without printing)
        sync_device()
        ram_usage = psutil.virtual_memory().used / 1e9
        gpu_allocated = 0
        gpu_reserved = 0
        
        if device.type == "mps":
            try:
                gpu_allocated = torch.mps.current_allocated_memory() / 1e9
                gpu_reserved = torch.mps.driver_allocated_memory() / 1e9
            except Exception:
                pass
        elif device.type == "cuda":
            gpu_allocated = torch.cuda.memory_allocated() / 1e9
            gpu_reserved = torch.cuda.memory_reserved() / 1e9
            
        # Get memory resources
        resources = {
            'ram': ram_usage,
            'gpu_allocated': gpu_allocated,
            'gpu_reserved': gpu_reserved
        }
        
        # Stop power monitoring and get statistics
        power_stats = power_monitor.stop_monitoring()
        
        # Store metrics
        metrics['train_acc'].append(train_accuracy)
        metrics['test_acc'].append(test_accuracy)
        metrics['epoch_times'].append(epoch_time)
        metrics['inference_times'].append(inference_time)
        metrics['ram_usages'].append(resources['ram'])
        metrics['gpu_allocated'].append(resources['gpu_allocated'])
        metrics['gpu_reserved'].append(resources['gpu_reserved'])
        metrics['epoch_metrics'].append(epoch_metrics)
        if power_stats:
            metrics['epoch_power_stats'].append(power_stats)
        
        # Plot confusion matrix for this epoch (without console output)
        plot_confusion_matrix(
            all_targets, all_predictions, 
            classes=gesture_class_names, 
            epoch=epoch,
            save_dir=results_dir
        )
        
        # Save per-epoch metrics to CSV (silently)
        epoch_df = pd.DataFrame({
            'Class': gesture_class_names,
            'Precision': epoch_metrics['class_precision'],
            'Recall': epoch_metrics['class_recall'],
            'F1 Score': epoch_metrics['class_f1']
        })
        epoch_df.to_csv(f"{results_dir}/metrics_epoch_{epoch+1}.csv", index=False)
        
        # Print epoch summary in consistent single-line format
        if device.type == "mps":
            memory_str = f"RAM: {ram_usage:.2f} GB, GPU Allocated: {gpu_allocated:.2f} GB, GPU Driver: {gpu_reserved:.2f} GB"
        elif device.type == "cuda":
            memory_str = f"RAM: {ram_usage:.2f} GB, GPU Allocated: {gpu_allocated:.2f} GB, GPU Reserved: {gpu_reserved:.2f} GB"
        else:
            memory_str = f"RAM: {ram_usage:.2f} GB"
            
        if power_stats:
            power_str = f"GPU Power: {power_stats['mean']:.2f} W"
        else:
            power_str = "GPU Power: N/A"
            
        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Acc: {train_accuracy:.2f}%, "
              f"Test Acc: {test_accuracy:.2f}%, "
              f"Macro F1: {epoch_metrics['macro_f1']:.4f}, "
              f"Time: {epoch_time:.2f}s, "
              f"Inference: {inference_time:.2f}s, "
              f"{memory_str}, {power_str}")
        
        # Force garbage collection to free memory
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        sync_device()
    
    return metrics

def evaluate(model, dataloader, criterion, detailed=False):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    
    # For confusion matrix
    all_predictions = []
    all_targets = []
    
    # Start timing
    start_time = time.time()
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            # Move to device with non_blocking for potential speedup
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Ensure tensors are on device before computation
            sync_device()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Store for metrics calculation
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
    
    # Ensure all operations are complete before stopping timer
    sync_device()
    inference_time = time.time() - start_time
    accuracy = 100 * correct / total
    
    if detailed:
        print(f"Detailed Test Results:")
        print(f"  Loss: {running_loss/len(dataloader):.4f}")
        print(f"  Accuracy: {accuracy:.2f}%")
        print(f"  Inference Time: {inference_time:.4f} seconds")
        
        print("  GPU Resource Usage During Testing:")
        log_memory_usage("Testing")
        
        # Calculate confusion matrix
        cm = confusion_matrix(all_targets, all_predictions)
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Print per-class accuracy
        per_class_acc = cm_percent.diagonal()
        for i in range(num_classes):
            class_name = gesture_class_names[i] if i < len(gesture_class_names) else f"Class {i}"
            print(f"  {class_name} Accuracy: {per_class_acc[i]:.2f}%")
    
    return accuracy, inference_time, all_targets, all_predictions

# ================= MAIN FUNCTION =================
def main():
    print("DVS Gesture Classification using snnTorch SNN with Cached Dataset")
    
    # Create results directory
    results_dir = "training_results_DVSGesture_SNN"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Try to force GPU to initialize fully
    if device.type == "mps":
        print("Initializing MPS device with a warm-up tensor operation...")
        dummy_input = torch.ones(1, 1, 1, 1, device=device)
        dummy_output = dummy_input * 2
        sync_device()
        print("MPS warm-up complete.")
    
    # Load datasets using cached implementation
    print("\nLoading DVSGesture dataset using cached approach...")
    train_dataset = CachedDVSGestureDataset(train=True, time_steps=time_steps)
    test_dataset = CachedDVSGestureDataset(train=False, time_steps=time_steps)
    
    # Use optimal number of workers for dataloader based on device
    num_workers = 2 if device.type == "mps" else 4
    
    # Create standard PyTorch dataloaders with the cached datasets
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Initial memory usage
    print("\nInitial memory usage:")
    log_memory_usage("Before Training")
    
    # Initialize the model, loss function, and optimizer
    model = DVSSNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train the model
    print("\nStarting training process...")
    history = train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs)
    
    # Save the model
    model_save_path = os.path.join(results_dir, 'dvs_gesture_snntorch_model.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    
    # Plot comprehensive training results
    plot_training_results(history, gesture_class_names)
    
    # Perform final evaluation
    print("\nPerforming final evaluation...")
    final_accuracy, inference_time, all_targets, all_predictions = evaluate(model, test_loader, criterion, detailed=True)
    final_metrics = calculate_metrics(all_targets, all_predictions, num_classes)

    plot_confusion_matrix(all_targets, all_predictions, classes=gesture_class_names, epoch=None, save_dir=results_dir)

    
    # Summary statistics in concise format
    print(f"\nTraining Summary:")
    print(f"Final Test Accuracy: {final_accuracy:.2f}%, Macro F1: {final_metrics['macro_f1']:.4f}")
    print(f"Total Training Time: {sum(history['epoch_times']):.2f}s")
    
    # Calculate and display power metrics
    power_metrics = calculate_power_metrics(history)
    if power_metrics:
        print(f"Peak Power: {power_metrics['peak_power']:.2f}W, Avg Power: {power_metrics['average_power']:.2f}W")
        print(f"Total Energy: {power_metrics['total_energy_wh']:.4f} Watt-hours")
    
    # Generate classification report
    report = classification_report(all_targets, all_predictions, 
                                  target_names=gesture_class_names, 
                                  digits=4)
    
    # Save the report to a file (without printing to console)
    with open(f"{results_dir}/final_classification_report.txt", "w") as f:
        f.write(report)
    
    # Create class performance dashboard
    create_class_performance_dashboard(
        all_targets, all_predictions, 
        classes=gesture_class_names, 
        results=history,
        results_dir=results_dir
    )
    
    # Calculate and display power metrics
    power_metrics = calculate_power_metrics(history)
    if power_metrics:
        print("\nDetailed Power Metrics:")
        print(f"Peak Power: {power_metrics['peak_power']:.2f} W")
        print(f"Average Power: {power_metrics['average_power']:.2f} W")
        print(f"Total Energy: {power_metrics['total_energy_joules']:.2f} Joules ({power_metrics['total_energy_wh']:.4f} Watt-hours)")
        
        if power_metrics['power_efficiency']:
            avg_efficiency = sum(power_metrics['power_efficiency']) / len(power_metrics['power_efficiency'])
            print(f"Average Energy per 1% Accuracy Improvement: {avg_efficiency:.2f} Joules ({avg_efficiency/3600:.4f} Wh)")
    
    print("Training completed. All results saved to 'training_results_DVSGesture_SNN' directory")
    
    # Final memory usage at the end
    print("Final memory usage:")
    log_memory_usage("After Training")

if __name__ == "__main__":
    main()