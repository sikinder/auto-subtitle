#!/usr/bin/env python3
"""
Performance test script for the optimized auto-subtitle tool.
This script helps measure the performance improvements of the parallel implementation.
"""

import time
import os
import sys
import subprocess
import argparse
from pathlib import Path


def create_test_video(output_path, duration=30):
    """Create a test video file using ffmpeg."""
    cmd = [
        'ffmpeg', '-f', 'lavfi', '-i', f'testsrc=duration={duration}:size=320x240:rate=1',
        '-f', 'lavfi', '-i', f'sine=frequency=1000:duration={duration}',
        '-c:v', 'libx264', '-c:a', 'aac', '-shortest', '-y', output_path
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error creating test video: {e}")
        return False


def run_auto_subtitle(video_files, output_dir, max_workers=None, model="tiny"):
    """Run auto_subtitle with specified parameters and measure time."""
    cmd = [
        sys.executable, '-m', 'auto_subtitle.cli',
        *video_files,
        '-o', output_dir,
        '--model', model,
        '--srt_only', 'true'
    ]
    
    if max_workers:
        cmd.extend(['--max_workers', str(max_workers)])
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        end_time = time.time()
        
        return {
            'success': True,
            'duration': end_time - start_time,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
    except subprocess.CalledProcessError as e:
        end_time = time.time()
        return {
            'success': False,
            'duration': end_time - start_time,
            'stdout': e.stdout,
            'stderr': e.stderr,
            'error': str(e)
        }


def main():
    parser = argparse.ArgumentParser(description='Test auto-subtitle performance')
    parser.add_argument('--num_videos', type=int, default=4, 
                       help='Number of test videos to create')
    parser.add_argument('--video_duration', type=int, default=30,
                       help='Duration of each test video in seconds')
    parser.add_argument('--model', default='tiny',
                       help='Whisper model to use for testing')
    parser.add_argument('--test_dir', default='./test_performance',
                       help='Directory for test files')
    
    args = parser.parse_args()
    
    # Create test directory
    test_dir = Path(args.test_dir)
    test_dir.mkdir(exist_ok=True)
    
    videos_dir = test_dir / 'videos'
    videos_dir.mkdir(exist_ok=True)
    
    output_dir = test_dir / 'output'
    output_dir.mkdir(exist_ok=True)
    
    print(f"Creating {args.num_videos} test videos...")
    
    # Create test videos
    video_files = []
    for i in range(args.num_videos):
        video_path = videos_dir / f'test_video_{i+1}.mp4'
        if create_test_video(str(video_path), args.video_duration):
            video_files.append(str(video_path))
            print(f"Created {video_path}")
        else:
            print(f"Failed to create {video_path}")
    
    if not video_files:
        print("No test videos were created. Exiting.")
        return
    
    print(f"\nTesting with {len(video_files)} videos...")
    
    # Test different worker configurations
    worker_configs = [1, 2, 4, None]  # None means auto-detect
    
    results = {}
    
    for workers in worker_configs:
        worker_label = f"{workers} workers" if workers else "auto workers"
        print(f"\n--- Testing with {worker_label} ---")
        
        result = run_auto_subtitle(
            video_files, 
            str(output_dir), 
            max_workers=workers,
            model=args.model
        )
        
        results[worker_label] = result
        
        if result['success']:
            print(f"✅ Completed in {result['duration']:.2f} seconds")
        else:
            print(f"❌ Failed after {result['duration']:.2f} seconds")
            print(f"Error: {result.get('error', 'Unknown error')}")
    
    # Print summary
    print("\n" + "="*50)
    print("PERFORMANCE SUMMARY")
    print("="*50)
    
    successful_results = {k: v for k, v in results.items() if v['success']}
    
    if successful_results:
        fastest = min(successful_results.items(), key=lambda x: x[1]['duration'])
        
        print(f"Fastest configuration: {fastest[0]} ({fastest[1]['duration']:.2f}s)")
        print("\nAll results:")
        
        for config, result in successful_results.items():
            speedup = fastest[1]['duration'] / result['duration']
            print(f"  {config}: {result['duration']:.2f}s (speedup: {speedup:.2f}x)")
    else:
        print("No successful runs to compare.")
    
    # Cleanup
    print(f"\nTest files are in: {test_dir}")
    print("You can delete the test directory when done.")


if __name__ == '__main__':
    main()
