# Zero-shot Object Detection using Grounding DINO
# Complete implementation for Project-5

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import torch
import torchvision.transforms as transforms
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import json
import requests
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

class GroundingDINODetector:
    """
    Zero-shot object detector using Grounding DINO model
    """
    
    def __init__(self, model_name="IDEA-Research/grounding-dino-base", device=None):
        """
        Initialize the Grounding DINO detector
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run the model on ('cuda', 'cpu', or None for auto)
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load model and processor
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        print("Grounding DINO model loaded successfully!")
    
    def detect_objects(self, image_path: str, prompt: str, 
                      confidence_threshold: float = 0.3) -> Dict:
        """
        Perform zero-shot object detection on an image
        
        Args:
            image_path: Path to the input image
            prompt: Natural language prompt describing objects to detect
            confidence_threshold: Minimum confidence score for detections
            
        Returns:
            Dictionary containing detection results
        """
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        
        # Process inputs
        inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Post-process results
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            box_threshold=confidence_threshold,
            text_threshold=confidence_threshold,
            target_sizes=[image.size[::-1]]  # (height, width)
        )[0]
        
        return {
            'image': image,
            'boxes': results['boxes'].cpu().numpy(),
            'scores': results['scores'].cpu().numpy(),
            'labels': results['labels'],
            'prompt': prompt,
            'image_path': image_path
        }
    
    def visualize_detections(self, detection_results: Dict, 
                           save_path: str = None, show_plot: bool = True):
        """
        Visualize detection results with bounding boxes
        
        Args:
            detection_results: Results from detect_objects()
            save_path: Path to save the visualization
            show_plot: Whether to display the plot
        """
        image = detection_results['image']
        boxes = detection_results['boxes']
        scores = detection_results['scores']
        labels = detection_results['labels']
        prompt = detection_results['prompt']
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image)
        
        # Draw bounding boxes
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            
            # Create rectangle patch
            rect = patches.Rectangle(
                (x1, y1), width, height,
                linewidth=2, edgecolor='red', facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add label and confidence score
            ax.text(
                x1, y1 - 10,
                f'{label}: {score:.2f}',
                fontsize=12,
                color='red',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8)
            )
        
        ax.set_title(f'Zero-shot Detection Results\nPrompt: "{prompt}"', fontsize=14)
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()

class DatasetManager:
    """
    Handle dataset preparation and image management
    """
    
    def __init__(self, dataset_dir: str = "test_images"):
        """
        Initialize dataset manager
        
        Args:
            dataset_dir: Directory to store test images
        """
        self.dataset_dir = dataset_dir
        os.makedirs(dataset_dir, exist_ok=True)
    
    def download_sample_images(self, num_images: int = 10) -> List[str]:
        """
        Download sample images for testing (placeholder URLs - replace with actual dataset)
        
        Args:
            num_images: Number of sample images to download
            
        Returns:
            List of image file paths
        """
        # Sample image URLs (replace with actual COCO or Open Images URLs)
        sample_urls = [
            "https://images.unsplash.com/photo-1546527868-ccb7ee7dfa6a?w=800",  # cat
            "https://images.unsplash.com/photo-1552053831-71594a27632d?w=800",  # dog
            "https://images.unsplash.com/photo-1484704849700-f032a568e944?w=800",  # laptop
            "https://images.unsplash.com/photo-1571019613454-1cb2f99b2d8b?w=800",  # bottle
            "https://images.unsplash.com/photo-1549298916-b41d501d3772?w=800",  # book
        ]
        
        image_paths = []
        for i, url in enumerate(sample_urls[:num_images]):
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    image_path = os.path.join(self.dataset_dir, f"sample_{i+1}.jpg")
                    with open(image_path, 'wb') as f:
                        f.write(response.content)
                    image_paths.append(image_path)
                    print(f"Downloaded: {image_path}")
            except Exception as e:
                print(f"Failed to download image {i+1}: {e}")
        
        return image_paths
    
    def get_test_prompts(self) -> Dict[str, List[str]]:
        """
        Define test prompts for different categories
        
        Returns:
            Dictionary of prompt categories and variations
        """
        return {
            "animals": [
                "cat", "dog", "bird", "fish", "horse",
                "a cat sitting", "brown dog", "small bird"
            ],
            "objects": [
                "laptop", "bottle", "book", "chair", "table",
                "water bottle", "laptop computer", "wooden chair"
            ],
            "vehicles": [
                "car", "truck", "bicycle", "motorcycle",
                "red car", "delivery truck", "mountain bike"
            ],
            "food": [
                "apple", "banana", "pizza", "sandwich",
                "fresh apple", "ripe banana", "cheese pizza"
            ]
        }

class ExperimentRunner:
    """
    Run comprehensive experiments and evaluations
    """
    
    def __init__(self, detector: GroundingDINODetector, dataset_manager: DatasetManager):
        """
        Initialize experiment runner
        
        Args:
            detector: GroundingDINODetector instance
            dataset_manager: DatasetManager instance
        """
        self.detector = detector
        self.dataset_manager = dataset_manager
        self.results = []
    
    def run_baseline_experiments(self, image_paths: List[str], 
                                prompts: List[str]) -> List[Dict]:
        """
        Run baseline zero-shot detection experiments
        
        Args:
            image_paths: List of image file paths
            prompts: List of prompts to test
            
        Returns:
            List of detection results
        """
        print("Running baseline experiments...")
        results = []
        
        for image_path in image_paths:
            for prompt in prompts:
                print(f"Processing: {os.path.basename(image_path)} with prompt: '{prompt}'")
                
                try:
                    result = self.detector.detect_objects(image_path, prompt)
                    result['experiment_type'] = 'baseline'
                    results.append(result)
                    
                    # Visualize results
                    vis_path = f"results/baseline_{os.path.basename(image_path).split('.')[0]}_{prompt.replace(' ', '_')}.jpg"
                    os.makedirs("results", exist_ok=True)
                    self.detector.visualize_detections(result, vis_path, show_plot=False)
                    
                except Exception as e:
                    print(f"Error processing {image_path} with '{prompt}': {e}")
        
        self.results.extend(results)
        return results
    
    def run_prompt_variability_experiments(self, image_paths: List[str]) -> List[Dict]:
        """
        Test different prompt formulations for the same object
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            List of detection results
        """
        print("Running prompt variability experiments...")
        
        # Define prompt variations for testing
        prompt_variations = {
            "cat": ["cat", "a cat", "feline", "kitten", "domestic cat", "cat animal"],
            "dog": ["dog", "a dog", "canine", "puppy", "domestic dog", "dog animal"],
            "laptop": ["laptop", "a laptop", "computer", "notebook computer", "laptop computer"]
        }
        
        results = []
        
        for image_path in image_paths:
            for base_object, variations in prompt_variations.items():
                print(f"\nTesting prompt variations for '{base_object}' on {os.path.basename(image_path)}")
                
                for prompt in variations:
                    try:
                        result = self.detector.detect_objects(image_path, prompt)
                        result['experiment_type'] = 'prompt_variability'
                        result['base_object'] = base_object
                        results.append(result)
                        
                        print(f"  '{prompt}': {len(result['boxes'])} detections")
                        
                    except Exception as e:
                        print(f"  Error with '{prompt}': {e}")
        
        self.results.extend(results)
        return results
    
    def analyze_results(self):
        """
        Analyze and summarize experimental results
        """
        if not self.results:
            print("No results to analyze!")
            return
        
        print("\n" + "="*50)
        print("RESULTS ANALYSIS")
        print("="*50)
        
        # Overall statistics
        total_experiments = len(self.results)
        successful_detections = sum(1 for r in self.results if len(r['boxes']) > 0)
        
        print(f"Total experiments: {total_experiments}")
        print(f"Successful detections: {successful_detections}")
        print(f"Success rate: {successful_detections/total_experiments:.2%}")
        
        # Analysis by experiment type
        baseline_results = [r for r in self.results if r['experiment_type'] == 'baseline']
        variability_results = [r for r in self.results if r['experiment_type'] == 'prompt_variability']
        
        if baseline_results:
            baseline_success = sum(1 for r in baseline_results if len(r['boxes']) > 0)
            print(f"\nBaseline experiments: {len(baseline_results)}")
            print(f"Baseline success rate: {baseline_success/len(baseline_results):.2%}")
        
        if variability_results:
            variability_success = sum(1 for r in variability_results if len(r['boxes']) > 0)
            print(f"Variability experiments: {len(variability_results)}")
            print(f"Variability success rate: {variability_success/len(variability_results):.2%}")
        
        # Confidence score analysis
        all_scores = []
        for result in self.results:
            all_scores.extend(result['scores'])
        
        if all_scores:
            print(f"\nConfidence Scores:")
            print(f"Mean: {np.mean(all_scores):.3f}")
            print(f"Std: {np.std(all_scores):.3f}")
            print(f"Min: {np.min(all_scores):.3f}")
            print(f"Max: {np.max(all_scores):.3f}")
        
        # Save results to JSON
        self.save_results_json()
    
    def save_results_json(self, filename: str = "results/experiment_results.json"):
        """
        Save results to JSON file for further analysis
        
        Args:
            filename: Path to save JSON file
        """
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Prepare results for JSON serialization
        json_results = []
        for result in self.results:
            json_result = {
                'image_path': result['image_path'],
                'prompt': result['prompt'],
                'experiment_type': result['experiment_type'],
                'num_detections': len(result['boxes']),
                'confidence_scores': result['scores'].tolist(),
                'labels': result['labels'],
                'boxes': result['boxes'].tolist()
            }
            if 'base_object' in result:
                json_result['base_object'] = result['base_object']
            json_results.append(json_result)
        
        with open(filename, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"Results saved to: {filename}")

# Main execution pipeline
def main():
    """
    Main function to run the complete pipeline
    """
    print("Zero-shot Object Detection using Grounding DINO")
    print("="*50)
    
    # Initialize components
    print("Initializing detector...")
    detector = GroundingDINODetector()
    
    print("Setting up dataset manager...")
    dataset_manager = DatasetManager()
    
    # Milestone 1: Prepare test dataset
    print("\nMilestone 1: Preparing test dataset...")
    image_paths = dataset_manager.download_sample_images(num_images=5)
    test_prompts = dataset_manager.get_test_prompts()
    
    if not image_paths:
        print("No images downloaded. Please add your own images to the 'test_images' directory.")
        return
    
    # Initialize experiment runner
    experiment_runner = ExperimentRunner(detector, dataset_manager)
    
    # Milestone 2: Baseline experiments
    print("\nMilestone 2: Running baseline experiments...")
    baseline_prompts = ["cat", "dog", "laptop", "bottle", "book"]
    baseline_results = experiment_runner.run_baseline_experiments(image_paths, baseline_prompts)
    
    # Milestone 3: Qualitative evaluation (visualizations already generated)
    print("\nMilestone 3: Qualitative evaluation completed (check 'results' directory)")
    
    # Milestone 4: Prompt variability experiments
    print("\nMilestone 4: Running prompt variability experiments...")
    variability_results = experiment_runner.run_prompt_variability_experiments(image_paths[:2])  # Use fewer images for variability
    
    # Final analysis
    print("\nFinal Analysis:")
    experiment_runner.analyze_results()
    
    print("\nPipeline completed successfully!")
    print("Check the 'results' directory for visualizations and analysis files.")

if __name__ == "__main__":
    main()