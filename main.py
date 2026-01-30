"""
Logo Detection Pipeline - Main Entry Point

This script provides a command-line interface for the logo detection system.
It supports processing single images or entire directories.

Usage:
    python main.py --image path/to/image.jpg
    python main.py --directory path/to/images/
    python main.py --help

Author: Team 4 - Computer Vision Bootcamp
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import DetectionPipeline, create_pipeline
from src import config


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Logo Detection Pipeline - Detect and visualize logos in images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Process a single image:
    python main.py --image data/input/logo.jpg

  Process all images in a directory:
    python main.py --directory data/input/

  Show detection results:
    python main.py --image data/input/logo.jpg --show

  Use custom confidence threshold:
    python main.py --image data/input/logo.jpg --confidence 0.7
        """
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "-i", "--image",
        type=str,
        help="Path to a single image file"
    )
    input_group.add_argument(
        "-d", "--directory",
        type=str,
        help="Path to a directory containing images"
    )
    input_group.add_argument(
        "-V", "--video",
        type=str,
        help="Path to a video file"
    )
    
    # Output options
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help=f"Output directory for annotated images (default: {config.OUTPUT_DIR})"
    )
    
    # Model options
    parser.add_argument(
        "-m", "--model",
        type=str,
        default=None,
        help=f"Path to YOLO model file (default: {config.DEFAULT_MODEL})"
    )
    
    parser.add_argument(
        "-c", "--confidence",
        type=float,
        default=None,
        help=f"Confidence threshold (0.0-1.0, default: {config.CONFIDENCE_THRESHOLD})"
    )
    
    # Display options
    parser.add_argument(
        "-s", "--show",
        action="store_true",
        help="Display detection results (only for single image)"
    )
    
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save annotated images"
    )
    
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Search subdirectories when processing a directory"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser.parse_args()


def main():
    """Main function to run the detection pipeline."""
    args = parse_arguments()
    
    # Validate confidence threshold
    if args.confidence is not None:
        if not 0.0 <= args.confidence <= 1.0:
            print("Error: Confidence must be between 0.0 and 1.0")
            sys.exit(1)
    
    # Create pipeline
    pipeline = create_pipeline(
        model=args.model,
        confidence=args.confidence
    )
    
    # Load model
    print("\n" + "="*60)
    print("  LOGO DETECTION PIPELINE")
    print("="*60 + "\n")
    
    if not pipeline.load_model():
        print("Error: Could not load model. Please check your installation.")
        sys.exit(1)
    
    # Determine output directory
    output_dir = Path(args.output) if args.output else config.OUTPUT_DIR
    
    # Process based on input type
    if args.image:
        # Single image processing
        image_path = Path(args.image)
        
        if not image_path.exists():
            print(f"Error: Image not found: {image_path}")
            sys.exit(1)
        
        print(f"Processing image: {image_path}")
        
        result = pipeline.process_image(
            image_path,
            save_output=not args.no_save,
            show_result=args.show,
            output_dir=output_dir
        )
        
        if result['success']:
            print(f"\n✅ Processing complete!")
            print(f"   Detections found: {result['detection_count']}")
            print(f"   Processing time: {result['processing_time']:.3f}s")
            
            if result['output_path']:
                print(f"   Output saved to: {result['output_path']}")
            
            # Print detection details
            if result['detections'] and args.verbose:
                print("\n   Detection details:")
                for i, det in enumerate(result['detections'], 1):
                    print(f"   {i}. {det['label']}: {det['confidence']:.1%}")
        else:
            print(f"\n❌ Processing failed: {result.get('error', 'Unknown error')}")
            sys.exit(1)
    
    elif args.video:
        # Video processing
        video_path = Path(args.video)
        
        if not video_path.exists():
            print(f"Error: Video not found: {video_path}")
            sys.exit(1)
        
        print(f"Processing video: {video_path}")
        
        result = pipeline.process_video(
            video_path,
            output_dir=output_dir,
            show_result=args.show
        )
        
        if result['success']:
            print(f"\n✅ Processing complete!")
            print(f"   Frames processed: {result['frame_count']}")
            print(f"   Total detections: {result['detection_count']}")
            print(f"   Processing time: {result['processing_time']:.2f}s")
            
            if result['output_path']:
                print(f"   Output saved to: {result['output_path']}")
        else:
            print(f"\n❌ Processing failed: {result.get('error', 'Unknown error')}")
            sys.exit(1)
    
    else:
        # Directory processing
        input_dir = Path(args.directory)
        
        if not input_dir.exists():
            print(f"Error: Directory not found: {input_dir}")
            sys.exit(1)
        
        print(f"Processing directory: {input_dir}")
        if args.recursive:
            print("(Including subdirectories)")
        
        results = pipeline.process_directory(
            input_dir,
            output_dir=output_dir,
            recursive=args.recursive,
            save_output=not args.no_save
        )
        
        if not results:
            print("\n⚠️ No images were processed")
            sys.exit(1)
        
        # Summary statistics
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]
        total_detections = sum(r.get('detection_count', 0) for r in successful)
        total_time = sum(r.get('processing_time', 0) for r in successful)
        
        print(f"\n✅ Batch processing complete!")
        print(f"   Successful: {len(successful)}/{len(results)}")
        print(f"   Total detections: {total_detections}")
        print(f"   Total processing time: {total_time:.2f}s")
        
        if failed:
            print(f"\n⚠️ Failed to process {len(failed)} image(s)")
            if args.verbose:
                for r in failed:
                    print(f"   - {r['input_path']}: {r.get('error', 'Unknown error')}")
    
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()
