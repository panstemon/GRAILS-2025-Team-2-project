import torch
import argparse
import pickle
import numpy as np
from ultralytics import YOLO

def evaluate_yolo(model_path, dataset_yaml, output_file):
    model = YOLO(model_path)  # Load YOLO model
    results = model.val(data=dataset_yaml, device = 'cuda' if torch.cuda.is_available() else 'cpu')
    
    # Save results object to a file
    with open(output_file, "wb") as f:
        pickle.dump(results, f)
    
    print(f"Results saved to {output_file}")
    print_metrics(results)

def print_metrics(results):
    metrics = {
        "Precision": np.mean(results.box.p),  # Average Precision
        "Recall": np.mean(results.box.r),  # Average Recall
        "mAP@0.5": results.box.map50,  # mAP at IoU 0.5
        "mAP@0.5:0.95": results.box.map  # mAP at IoU range 0.5-0.95
    }
    
    # Print the averaged metrics
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

def analyze_pkl(pkl_path):
    with open(pkl_path, "rb") as f:
        results = pickle.load(f)
    print_metrics(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO Evaluation Script")
    parser.add_argument("--version", type=str, required=True, help="YOLO version (e.g., 5n, 6s)")
    parser.add_argument("--dataset", type=str, default="./coco.yaml", help="Path to dataset YAML file")
    parser.add_argument("--analyze", type=str, help="Path to .pkl file to analyze results")
    
    args = parser.parse_args()
    
    if args.analyze:
        analyze_pkl(args.analyze)
    else:
        yolo_path = f"./models/yolov{args.version}.pt"
        output_file = f"./metrics/evaluation_results/yolo{args.version}_evaluation_results.pkl"
        evaluate_yolo(yolo_path, args.dataset, output_file)
