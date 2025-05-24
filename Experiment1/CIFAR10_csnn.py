#!/usr/bin/env python
# coding: utf-8

# ## Step 1: Import Libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import time
import matplotlib.pyplot as plt
import subprocess
import re
import psutil
import os
import threading
import numpy as np
from datetime import datetime
import platform
import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import utils
from snntorch import spikegen

# IMPORTS for metrics and visualization
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, classification_report
import seaborn as sns
import pandas as pd


# Class to handle continuous power monitoring in a separate thread
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
                    power_output = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode("utf-8")
                    
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


# Function to log memory usage
def log_memory_usage(stage=""):
    """Logs RAM and GPU memory usage for a given training stage and returns a formatted string."""
    ram_usage = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    memory_log = f"[{stage}] RAM Usage: {ram_usage:.2f} MB"

    if device == "mps":
        # For Apple Silicon GPUs
        gpu_alloc = torch.mps.current_allocated_memory() / (1024 * 1024)
        gpu_driver = torch.mps.driver_allocated_memory() / (1024 * 1024)
        memory_log += f", GPU Allocated Memory: {gpu_alloc:.2f} MB, GPU Driver Memory: {gpu_driver:.2f} MB"
    elif device == "cuda":
        # For NVIDIA GPUs
        gpu_alloc = torch.cuda.memory_allocated() / (1024 * 1024)
        gpu_reserved = torch.cuda.memory_reserved() / (1024 * 1024)
        memory_log += f", GPU Allocated Memory: {gpu_alloc:.2f} MB, GPU Reserved Memory: {gpu_reserved:.2f} MB"

    return memory_log


# Calculate and plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes, epoch=None, save_dir='.'):
    """
    Calculate and plot confusion matrix
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        classes: List of class names
        epoch: Epoch number (None for final evaluation)
        save_dir: Directory to save the plot
    """
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


# Calculate classification metrics
def calculate_metrics(y_true, y_pred, num_classes=10):
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


# Plot metrics per class over epochs
def plot_metrics_per_class(results, classes, results_dir="training_results_CIFAR10SNN"):
    """Plot precision, recall, and F1 score for each class over epochs."""
    num_epochs = len(results["epoch_metrics"])
    num_classes = len(classes) if classes else len(results["class_f1_history"][0])
    
    # Plot F1 score for each class over epochs
    plt.figure(figsize=(12, 8))
    for i in range(num_classes):
        class_f1 = [epoch_f1[i] for epoch_f1 in results["class_f1_history"]]
        plt.plot(range(1, num_epochs+1), class_f1, 
                marker='o', label=classes[i] if classes else f"Class {i}")
    
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
    plt.plot(range(1, num_epochs+1), results["macro_precision_history"], 
            'b-', marker='o', label='Macro Precision')
    plt.plot(range(1, num_epochs+1), results["macro_recall_history"], 
            'g-', marker='s', label='Macro Recall')
    plt.plot(range(1, num_epochs+1), results["macro_f1_history"], 
            'r-', marker='^', label='Macro F1')
    
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Overall Metrics Over Time')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{results_dir}/overall_metrics.png")
    plt.close()


# Create integrated class performance visualization
def create_class_performance_dashboard(y_true, y_pred, classes, results, results_dir="training_results_CIFAR10SNN"):
    """
    Create an integrated visualization showing confusion matrix alongside class metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        classes: List of class names
        results: Training results dictionary
        results_dir: Directory to save the visualization
    """
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
    epochs = range(1, len(results["test_accuracies"]) + 1)
    
    # Select a few interesting classes (classes with highest/lowest accuracy)
    class_accuracies = np.diag(cm_percent)
    best_class_idx = np.argmax(class_accuracies)
    worst_class_idx = np.argmin(class_accuracies)
    
    # Plot F1 score over epochs for best and worst classes
    best_class_f1 = [epoch_f1[best_class_idx] for epoch_f1 in results["class_f1_history"]]
    worst_class_f1 = [epoch_f1[worst_class_idx] for epoch_f1 in results["class_f1_history"]]
    
    ax4.plot(epochs, best_class_f1, 'g-', marker='o', label=f'Best: {classes[best_class_idx]}')
    ax4.plot(epochs, worst_class_f1, 'r-', marker='s', label=f'Worst: {classes[worst_class_idx]}')
    ax4.plot(epochs, results["macro_f1_history"], 'b-', marker='^', label='Macro F1')
    
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('F1 Score')
    ax4.set_title('F1 Score Evolution')
    ax4.grid(True)
    ax4.legend()
    
    # Add an overall title
    plt.suptitle('Spiking Neural Network Performance on CIFAR-10: Class-Level Analysis', fontsize=16)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Adjust for main title
    plt.savefig(f"{results_dir}/class_performance_dashboard.png", dpi=300)
    plt.close()
    
    print(f"Class performance dashboard saved to {results_dir}/class_performance_dashboard.png")


# Modify the existing plot_training_results function to include memory graphs
def plot_training_results(results, classes=None):
    """Plot comprehensive training results including power, metrics, and memory."""
    results_dir = "training_results_CIFAR10SNN"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
    num_epochs = len(results["train_accuracies"])
    
    # Create figure with subplots - Adding a 5th subplot for memory
    fig, axs = plt.subplots(5, 1, figsize=(12, 24), 
                          gridspec_kw={'height_ratios': [1, 1, 1, 2, 1]})
    
    # Plot 1: Accuracy
    axs[0].plot(range(1, num_epochs+1), results["train_accuracies"], 'b-', label='Train Accuracy')
    axs[0].plot(range(1, num_epochs+1), results["test_accuracies"], 'r-', label='Test Accuracy')
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
    axs[2].plot(range(1, num_epochs+1), results["macro_precision_history"], 'b-', label='Precision')
    axs[2].plot(range(1, num_epochs+1), results["macro_recall_history"], 'g-', label='Recall')
    axs[2].plot(range(1, num_epochs+1), results["macro_f1_history"], 'r-', label='F1 Score')
    axs[2].set_xlabel('Epoch')
    axs[2].set_ylabel('Score')
    axs[2].set_title('Overall Metrics Over Time')
    axs[2].legend()
    axs[2].grid(True)
    
    # Plot 4: Power Profile
    if results["power_monitor"] and results["power_monitor"].get_power_profile():
        times, powers = results["power_monitor"].get_power_profile()
        axs[3].plot(times, powers, 'r-')
        axs[3].set_xlabel('Time (s)')
        axs[3].set_ylabel('Power (W)')
        axs[3].set_title('GPU Power Consumption Profile')
        
        # Add horizontal lines for min, mean, max power
        if results["epoch_power_stats"]:
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
    
    # Plot 5: Memory Usage
    if results.get("ram_usage") and results.get("gpu_allocated_memory"):
        axs[4].plot(range(1, num_epochs+1), results["ram_usage"], 'b-o', label='RAM Usage (MB)')
        if any(m > 0 for m in results["gpu_allocated_memory"]):
            axs[4].plot(range(1, num_epochs+1), results["gpu_allocated_memory"], 'r-s', label='GPU Allocated (MB)')
        
        # Show GPU driver memory only if it varies
        gpu_driver = results.get("gpu_driver_memory", [])
        if gpu_driver and max(gpu_driver) - min(gpu_driver) > 1:
            axs[4].plot(range(1, num_epochs+1), gpu_driver, 'g-^', label='GPU Driver (MB)')
        
        axs[4].set_xlabel('Epoch')
        axs[4].set_ylabel('Memory (MB)')
        axs[4].set_title('Memory Usage During Training')
        axs[4].legend()
        axs[4].grid(True)
    else:
        axs[4].text(0.5, 0.5, 'No memory data available', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=axs[4].transAxes)
    
    plt.tight_layout()
    plt.savefig(f'{results_dir}/comprehensive_training_results.png')
    plt.close()
    
    # Also create separate, more detailed plots
    
    # Detailed power profile
    if results["power_monitor"] and results["power_monitor"].get_power_profile():
        # (existing power profile code...)
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
    
    # Create detailed memory usage plots
    if results.get("ram_usage") and results.get("gpu_allocated_memory"):
        plot_memory_usage(results, results_dir)
    
    # If classes are available, create per-class metrics visualization
    if classes:
        plot_metrics_per_class(results, classes, results_dir)


# Add a new function to plot memory usage
def plot_memory_usage(results, results_dir="training_results_CIFAR10SNN"):
    """Plot memory usage data from training results."""
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Check if we have memory data
    if not results.get("ram_usage") or not results.get("gpu_allocated_memory"):
        print("No memory usage data available for plotting")
        return
    
    num_epochs = len(results["ram_usage"])
    epochs = range(1, num_epochs + 1)
    
    # Create figure for memory usage
    plt.figure(figsize=(12, 6))
    
    # Plot RAM usage
    plt.plot(epochs, results["ram_usage"], 'b-o', label='RAM Usage (MB)')
    
    # Plot GPU allocated memory if available
    if any(m > 0 for m in results["gpu_allocated_memory"]):
        plt.plot(epochs, results["gpu_allocated_memory"], 'r-s', label='GPU Allocated Memory (MB)')
    
    # Plot GPU driver/reserved memory if it varies
    gpu_driver_memory = results["gpu_driver_memory"]
    if gpu_driver_memory and max(gpu_driver_memory) - min(gpu_driver_memory) > 1:
        plt.plot(epochs, gpu_driver_memory, 'g-^', label='GPU Driver Memory (MB)')
    
    plt.xlabel('Epoch')
    plt.ylabel('Memory (MB)')
    plt.title('Memory Usage During Training')
    plt.grid(True)
    plt.legend()
    
    # Add annotations for exact values
    for i, (ram, gpu) in enumerate(zip(results["ram_usage"], results["gpu_allocated_memory"])):
        plt.annotate(f"{ram:.1f}", (i+1, ram), textcoords="offset points", 
                   xytext=(0,10), ha='center')
        if gpu > 0:
            plt.annotate(f"{gpu:.1f}", (i+1, gpu), textcoords="offset points", 
                       xytext=(0,-15), ha='center')
    
    plt.tight_layout()
    plt.savefig(f'{results_dir}/memory_usage.png')
    plt.close()
    
    print(f"Memory usage plot saved to {results_dir}/memory_usage.png")
    
    # Create detailed memory growth visualization
    plt.figure(figsize=(14, 8))
    
    # Create subplots
    gs = plt.GridSpec(2, 1, height_ratios=[2, 1])
    
    # Memory usage subplot
    ax1 = plt.subplot(gs[0])
    ax1.plot(epochs, results["ram_usage"], 'b-o', label='RAM Usage (MB)')
    
    if any(m > 0 for m in results["gpu_allocated_memory"]):
        ax1.plot(epochs, results["gpu_allocated_memory"], 'r-s', label='GPU Allocated Memory (MB)')
    
    # Add RAM annotations
    for i, ram in enumerate(results["ram_usage"]):
        ax1.annotate(f"{ram:.1f}", (i+1, ram), textcoords="offset points", 
                   xytext=(0,10), ha='center', fontsize=9)
    
    ax1.set_ylabel('Memory (MB)')
    ax1.set_title('Absolute Memory Usage')
    ax1.grid(True)
    ax1.legend()
    
    # Memory growth subplot (percentage increase from first epoch)
    ax2 = plt.subplot(gs[1])
    
    # Calculate percentage growth from first epoch
    ram_growth = [(ram / results["ram_usage"][0] - 1) * 100 for ram in results["ram_usage"]]
    ax2.plot(epochs, ram_growth, 'b-o', label='RAM Growth (%)')
    
    if any(m > 0 for m in results["gpu_allocated_memory"]):
        first_gpu = max(0.1, results["gpu_allocated_memory"][0])  # Avoid division by zero
        gpu_growth = [(gpu / first_gpu - 1) * 100 for gpu in results["gpu_allocated_memory"]]
        ax2.plot(epochs, gpu_growth, 'r-s', label='GPU Memory Growth (%)')
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Growth (%)')
    ax2.set_title('Memory Growth (% increase from first epoch)')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f'{results_dir}/memory_growth.png')
    plt.close()
    
    print(f"Memory growth plot saved to {results_dir}/memory_growth.png")


# Calculate comprehensive energy and power metrics (unchanged)
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
            accuracy_improvement = results["test_accuracies"][i] - results["test_accuracies"][i-1]
            if accuracy_improvement > 0:
                efficiency = stats["total_energy"] / accuracy_improvement
                metrics["power_efficiency"].append(efficiency)
    
    # Update metrics dictionary
    metrics["peak_power"] = max(peak_powers) if peak_powers else 0
    metrics["average_power"] = sum(mean_powers) / len(mean_powers) if mean_powers else 0
    metrics["total_energy_joules"] = total_energy
    metrics["total_energy_wh"] = total_energy / 3600  # Convert joules to watt-hours
    
    return metrics


# Modify the train_model function to track memory usage during training

def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs, classes=None):
    """
    Train the model with comprehensive metrics including memory tracking.
    
    Args:
        model: The PyTorch model to train
        train_loader: DataLoader for training data
        test_loader: DataLoader for testing data
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        num_epochs: Number of epochs to train for
        classes: List of class names
    
    Returns:
        dict: Training statistics
    """
    print(log_memory_usage("Before Training"))
    
    power_monitor = PowerMonitor(sample_interval=0.5)  # Sample every 0.5 seconds
    
    # Create results directory if it doesn't exist
    results_dir = "training_results_CIFAR10SNN"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Initialize metric tracking
    train_accuracies = []
    test_accuracies = []
    epoch_times = []
    epoch_power_stats = []
    losses = []  # Track losses
    
    # Track memory usage
    ram_usage = []
    gpu_allocated_memory = []
    gpu_driver_memory = []
    
    # Track precision, recall, and F1 scores for each epoch
    epoch_metrics = []
    
    # Create lists to store metrics for each class over epochs
    class_precision_history = []
    class_recall_history = []
    class_f1_history = []
    
    # Track macro and weighted averages over epochs
    macro_precision_history = []
    macro_recall_history = []
    macro_f1_history = []
    
    for epoch in range(num_epochs):
        start_time = time.time()  # Start timer for epoch
        
        # Start power monitoring for this epoch
        power_monitor.start_monitoring()
        
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
            if i % 100 == 99:  # Print every 100 mini-batches
                batch_loss = running_loss / 100
                losses.append(batch_loss)  # Record loss
                print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {batch_loss:.4f}")
                running_loss = 0.0
        
        train_accuracy = 100 * correct_train / total_train
        train_accuracies.append(train_accuracy)
        
        # Testing phase
        model.eval()
        correct_test = 0
        total_test = 0
        inference_start_time = time.time()
        
        # Collect all predictions and true labels for the test set
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                
                # Store for metrics calculation
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
                
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()
        
        inference_time = time.time() - inference_start_time
        test_accuracy = 100 * correct_test / total_test
        test_accuracies.append(test_accuracy)
        
        # Calculate and plot confusion matrix for this epoch
        if classes:
            cm = plot_confusion_matrix(
                all_targets, all_predictions, 
                classes=classes, 
                epoch=epoch,
                save_dir=results_dir
            )
        
        # Calculate precision, recall, and F1 score for this epoch
        metrics = calculate_metrics(all_targets, all_predictions)
        epoch_metrics.append(metrics)
        
        # Store metrics history
        class_precision_history.append(metrics['class_precision'])
        class_recall_history.append(metrics['class_recall'])
        class_f1_history.append(metrics['class_f1'])
        
        macro_precision_history.append(metrics['macro_precision'])
        macro_recall_history.append(metrics['macro_recall'])
        macro_f1_history.append(metrics['macro_f1'])
        
        # Stop power monitoring and get statistics
        power_stats = power_monitor.stop_monitoring()
        if power_stats:
            epoch_power_stats.append(power_stats)
        
        # Calculate epoch time
        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)
        
        # Track memory usage
        # Get current RAM usage
        current_ram = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        ram_usage.append(current_ram)
        
        # Get GPU memory stats
        if device == "mps":
            # For Apple Silicon GPUs
            gpu_alloc = torch.mps.current_allocated_memory() / (1024 * 1024)
            gpu_driver = torch.mps.driver_allocated_memory() / (1024 * 1024)
            gpu_allocated_memory.append(gpu_alloc)
            gpu_driver_memory.append(gpu_driver)
        elif device == "cuda":
            # For NVIDIA GPUs
            gpu_alloc = torch.cuda.memory_allocated() / (1024 * 1024)
            gpu_reserved = torch.cuda.memory_reserved() / (1024 * 1024)
            gpu_allocated_memory.append(gpu_alloc)
            gpu_driver_memory.append(gpu_reserved)
        else:
            # CPU only case
            gpu_allocated_memory.append(0)
            gpu_driver_memory.append(0)
        
        # Log memory usage
        memory_log = log_memory_usage("During Training")
        
        # Get GPU power reading
        gpu_power_log = ""
        if power_stats:
            gpu_power_log = f"GPU Power Consumption: {power_stats['mean']:.2f} W"
        
        # Print all info
        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Train Accuracy: {train_accuracy:.2f}%, "
              f"Test Accuracy: {test_accuracy:.2f}%, "
              f"Macro F1: {metrics['macro_f1']:.4f}, "
              f"Time: {epoch_time:.2f}s, "
              f"Total inference time: {inference_time:.2f} seconds, "
              f"{memory_log} {gpu_power_log}")
        
        # Step the learning rate scheduler
        scheduler.step()
        
        # Save per-epoch metrics to CSV
        epoch_df = pd.DataFrame({
            'Class': classes if classes else range(len(metrics['class_precision'])),
            'Precision': metrics['class_precision'],
            'Recall': metrics['class_recall'],
            'F1 Score': metrics['class_f1']
        })
        epoch_df.to_csv(f"{results_dir}/metrics_epoch_{epoch+1}.csv", index=False)
        
    
    # Generate detailed final classification report
    if classes:
        # This will print a full report to console
        print("\nDetailed Classification Report (Final Model):")
        report = classification_report(all_targets, all_predictions, 
                                      target_names=classes, 
                                      digits=4)
        print(report)
        
        # Save the report to a file
        with open(f"{results_dir}/final_classification_report.txt", "w") as f:
            f.write(report)
    
    # Compile results
    results = {
        "train_accuracies": train_accuracies,
        "test_accuracies": test_accuracies,
        "epoch_times": epoch_times,
        "epoch_power_stats": epoch_power_stats,
        "power_monitor": power_monitor,  # Keep for power profile plotting
        "losses": losses,
        
        # Store detailed metrics
        "epoch_metrics": epoch_metrics,
        "class_precision_history": class_precision_history,
        "class_recall_history": class_recall_history,
        "class_f1_history": class_f1_history,
        "macro_precision_history": macro_precision_history,
        "macro_recall_history": macro_recall_history,
        "macro_f1_history": macro_f1_history,
        
        # Store memory usage data
        "ram_usage": ram_usage,
        "gpu_allocated_memory": gpu_allocated_memory,
        "gpu_driver_memory": gpu_driver_memory
    }
    
    return results


# Setup device
device = (
    "mps" if torch.backends.mps.is_available() else  # Apple Silicon
    "cuda" if torch.cuda.is_available() else         # NVIDIA GPU
    "cpu"                                            # Fallback to CPU
)
print(f"Using device: {device}")

# CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
              'dog', 'frog', 'horse', 'ship', 'truck']

# Load and Preprocess Data
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # CIFAR-10 specific means and stds
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

train_dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
test_dataset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)

# Smaller batch size for SNN 
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)


# use fewer time steps for speed, but still leverage spiking neurons
num_steps = 3  # Reduce time steps for faster training
beta = 0.95  # Neuron decay rate

# Use a more stable surrogate gradient function
spike_grad = surrogate.fast_sigmoid(slope=10)


class SpikingConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(SpikingConvNet, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=False)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=False)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=False)
        
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        self.lif4 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=False)
        
        # Final layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Initialize membrane states
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()
        
        # Record output spikes for each step
        spk_out = []
        
        for step in range(num_steps):
            # First layer
            cur1 = self.bn1(self.conv1(x))
            spk1, mem1 = self.lif1(cur1, mem1)
            
            # Second layer
            cur2 = self.bn2(self.conv2(spk1))
            spk2, mem2 = self.lif2(cur2, mem2)
            
            # Third layer
            cur3 = self.bn3(self.conv3(spk2))
            spk3, mem3 = self.lif3(cur3, mem3)
            
            # Fourth layer
            cur4 = self.bn4(self.conv4(spk3))
            spk4, mem4 = self.lif4(cur4, mem4)
            
            # Pooling and readout
            x_pool = self.avgpool(spk4)
            x_flat = x_pool.view(batch_size, -1)
            out = self.fc(x_flat)
            
            spk_out.append(out)
        
        # Return average output over time steps
        return torch.stack(spk_out).mean(dim=0)


# Create a new instance of the SpikingConvNet
model = SpikingConvNet(num_classes=10).to(device)

# Define Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
# Different learning rate for SNN
optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)  
scheduler = StepLR(optimizer, step_size=10, gamma=0.3)  # Slower decay

# Additional helper function for debugging
def print_model_output_stats(outputs, msg=""):
    """Print statistics of model outputs to help debug training issues"""
    if isinstance(outputs, torch.Tensor):
        print(f"{msg} - Shape: {outputs.shape}, Min: {outputs.min().item():.4f}, "
              f"Max: {outputs.max().item():.4f}, Mean: {outputs.mean().item():.4f}")


# Train the model with improved metrics tracking
num_epochs = 30
training_results = train_model(
    model=model,
    train_loader=train_loader,
    test_loader=test_loader,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    num_epochs=num_epochs,
    classes=class_names  # NEW: Pass class names
)

# Log final memory usage
print(log_memory_usage("After Training"))

# Calculate comprehensive metrics
power_metrics = calculate_power_metrics(training_results)

# Compute final power metrics
if power_metrics:
    avg_gpu_power = power_metrics["average_power"]    
    print(f"\nAverage GPU Power Consumption: {avg_gpu_power:.2f} W")

# Log Final Metrics
total_training_time = sum(training_results["epoch_times"])
final_test_accuracy = training_results["test_accuracies"][-1]
final_f1_score = training_results["macro_f1_history"][-1]  # NEW

print(f"\nFinal Test Accuracy: {final_test_accuracy:.2f}%")
print(f"Final Macro F1 Score: {final_f1_score:.4f}")  # NEW
print(f"Total Training Time: {total_training_time:.2f}s")

# Additional comprehensive power metrics
if power_metrics:
    print("\nDetailed Power Metrics:")
    print(f"Peak Power: {power_metrics['peak_power']:.2f} W")
    print(f"Total Energy: {power_metrics['total_energy_wh']:.4f} Watt-hours")
    
    if power_metrics['power_efficiency']:
        avg_efficiency = sum(power_metrics['power_efficiency']) / len(power_metrics['power_efficiency'])
        print(f"Average Energy per 1% Accuracy Improvement: {avg_efficiency/3600:.4f} Wh")

# Plot comprehensive training results
plot_training_results(training_results, class_names)  # NEW: Pass class names

# Generate and save final test set confusion matrix
print("\nGenerating final detailed confusion matrix...")

# Run final evaluation on test set to collect predictions
model.eval()
all_predictions = []
all_targets = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        
        all_predictions.extend(predicted.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())

# Create and save final confusion matrix
cm = plot_confusion_matrix(
    all_targets, all_predictions, 
    classes=class_names, 
    epoch=None,  # None indicates this is the final matrix
    save_dir="training_results_CIFAR10SNN"
)

# Calculate normalized confusion matrix (by row)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Create per-class accuracy table
results_dir = "training_results_CIFAR10SNN"
class_accuracy = np.diag(cm_norm) * 100
class_df = pd.DataFrame({
    'Class': class_names,
    'Accuracy (%)': class_accuracy,
    'Support': np.sum(cm, axis=1)
})

# Add final metrics from the last epoch
final_metrics = training_results["epoch_metrics"][-1]
class_df['Precision'] = final_metrics['class_precision']
class_df['Recall'] = final_metrics['class_recall']
class_df['F1 Score'] = final_metrics['class_f1']

# Save class performance table
class_df.to_csv(f"{results_dir}/final_class_performance.csv", index=False)
print(f"\nPer-class performance metrics saved to {results_dir}/final_class_performance.csv")

# Print a summary table to console
print("\nPer-class Performance Summary:")
print(class_df.to_string(index=False))

# Create integrated class performance dashboard
create_class_performance_dashboard(
    all_targets, all_predictions,
    classes=class_names,
    results=training_results, 
    results_dir=results_dir
)

# Plot original loss curve as before
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(training_results["losses"])
plt.xlabel("Iterations (x100)")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.grid()

# Plot train/test accuracy
plt.subplot(1, 3, 2)
plt.plot(range(1, len(training_results["train_accuracies"]) + 1), training_results["train_accuracies"], label="Train Accuracy")
plt.plot(range(1, len(training_results["test_accuracies"]) + 1), training_results["test_accuracies"], label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Train/Test Accuracy")
plt.legend()
plt.grid()

# Plot epoch times
plt.subplot(1, 3, 3)
plt.plot(range(1, len(training_results["epoch_times"]) + 1), training_results["epoch_times"], label="Epoch Time", color="red")
plt.xlabel("Epoch")
plt.ylabel("Time (s)")
plt.title("Epoch Time")
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig(f'{results_dir}/snn_results.png')
plt.close()

# Save model
torch.save(model.state_dict(), f'{results_dir}/snn_cifar10_model.pth')
print(f"\nModel saved to {results_dir}/snn_cifar10_model.pth")

print("\nAll result visualizations have been saved to the 'training_results_CIFAR10SNN' directory.")
