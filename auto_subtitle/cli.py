import os
import ffmpeg
import whisper
import argparse
import warnings
import tempfile
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import partial
import psutil
from tqdm import tqdm
from .utils import filename, str2bool, write_srt


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("video", nargs="+", type=str,
                        help="paths to video files to transcribe")
    parser.add_argument("--model", default="small",
                        choices=whisper.available_models(), help="name of the Whisper model to use")
    parser.add_argument("--output_dir", "-o", type=str,
                        default=".", help="directory to save the outputs")
    parser.add_argument("--output_srt", type=str2bool, default=False,
                        help="whether to output the .srt file along with the video files")
    parser.add_argument("--srt_only", type=str2bool, default=False,
                        help="only generate the .srt file and not create overlayed video")
    parser.add_argument("--verbose", type=str2bool, default=False,
                        help="whether to print out the progress and debug messages")

    parser.add_argument("--task", type=str, default="transcribe", choices=[
                        "transcribe", "translate"], help="whether to perform X->X speech recognition ('transcribe') or X->English translation ('translate')")
    parser.add_argument("--language", type=str, default="auto", choices=["auto","af","am","ar","as","az","ba","be","bg","bn","bo","br","bs","ca","cs","cy","da","de","el","en","es","et","eu","fa","fi","fo","fr","gl","gu","ha","haw","he","hi","hr","ht","hu","hy","id","is","it","ja","jw","ka","kk","km","kn","ko","la","lb","ln","lo","lt","lv","mg","mi","mk","ml","mn","mr","ms","mt","my","ne","nl","nn","no","oc","pa","pl","ps","pt","ro","ru","sa","sd","si","sk","sl","sn","so","sq","sr","su","sv","sw","ta","te","tg","th","tk","tl","tr","tt","uk","ur","uz","vi","yi","yo","zh"],
    help="What is the origin language of the video? If unset, it is detected automatically.")

    parser.add_argument("--max_workers", type=int, default=None,
                        help="Maximum number of worker processes to use for parallel processing. If not set, uses number of CPU cores.")
    parser.add_argument("--threads_per_worker", type=int, default=2,
                        help="Number of threads per worker for I/O operations like audio extraction and video encoding.")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Number of videos to process in each batch. Larger batches use more memory but may be more efficient.")

    args = parser.parse_args().__dict__
    model_name: str = args.pop("model")
    output_dir: str = args.pop("output_dir")
    output_srt: bool = args.pop("output_srt")
    srt_only: bool = args.pop("srt_only")
    language: str = args.pop("language")
    max_workers: int = args.pop("max_workers")
    threads_per_worker: int = args.pop("threads_per_worker")
    batch_size: int = args.pop("batch_size")

    # Set default max_workers based on CPU cores if not specified
    if max_workers is None:
        max_workers = min(multiprocessing.cpu_count(), len(args["video"]))

    print(f"Using {max_workers} worker processes with {threads_per_worker} threads each")
    
    os.makedirs(output_dir, exist_ok=True)

    if model_name.endswith(".en"):
        warnings.warn(
            f"{model_name} is an English-only model, forcing English detection.")
        args["language"] = "en"
    # if translate task used and language argument is set, then use it
    elif language != "auto":
        args["language"] = language
        
    video_paths = args.pop("video")

    # Process videos in parallel
    process_videos_parallel(
        video_paths=video_paths,
        model_name=model_name,
        output_dir=output_dir,
        output_srt=output_srt,
        srt_only=srt_only,
        max_workers=max_workers,
        threads_per_worker=threads_per_worker,
        transcribe_args=args
    )


def extract_single_audio(video_path):
    """Extract audio from a single video file."""
    temp_dir = tempfile.gettempdir()
    output_path = os.path.join(temp_dir, f"{filename(video_path)}.wav")

    try:
        ffmpeg.input(video_path).output(
            output_path,
            acodec="pcm_s16le", ac=1, ar="16k"
        ).run(quiet=True, overwrite_output=True)
        return video_path, output_path
    except Exception as e:
        print(f"Error extracting audio from {filename(video_path)}: {e}")
        return video_path, None


def get_audio_parallel(paths, max_threads=4):
    """Extract audio from multiple videos in parallel."""
    audio_paths = {}

    print(f"Extracting audio from {len(paths)} videos using {max_threads} threads...")

    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        # Submit all tasks
        future_to_path = {executor.submit(extract_single_audio, path): path for path in paths}

        # Process completed tasks with progress bar
        with tqdm(total=len(paths), desc="Audio extraction") as pbar:
            for future in as_completed(future_to_path):
                video_path, audio_path = future.result()
                if audio_path:
                    audio_paths[video_path] = audio_path
                pbar.update(1)

    return audio_paths


def get_audio(paths):
    """Legacy function for backward compatibility."""
    return get_audio_parallel(paths, max_threads=4)


def transcribe_single_audio(args_tuple):
    """Transcribe a single audio file. Used for multiprocessing."""
    audio_path, video_path, model_name, output_srt, output_dir, transcribe_args = args_tuple

    try:
        # Load model in each process to avoid sharing issues
        model = whisper.load_model(model_name)

        srt_path = output_dir if output_srt else tempfile.gettempdir()
        srt_path = os.path.join(srt_path, f"{filename(video_path)}.srt")

        warnings.filterwarnings("ignore")
        result = model.transcribe(audio_path, **transcribe_args)
        warnings.filterwarnings("default")

        with open(srt_path, "w", encoding="utf-8") as srt:
            write_srt(result["segments"], file=srt)

        # Clean up model to free memory
        del model

        return video_path, srt_path
    except Exception as e:
        print(f"Error transcribing {filename(video_path)}: {e}")
        return video_path, None


def get_subtitles_parallel(audio_paths: dict, output_srt: bool, output_dir: str,
                          model_name: str, transcribe_args: dict, max_workers: int = 2):
    """Generate subtitles for multiple audio files in parallel."""
    subtitles_path = {}

    # Prepare arguments for each transcription task
    transcribe_tasks = []
    for video_path, audio_path in audio_paths.items():
        transcribe_tasks.append((
            audio_path, video_path, model_name, output_srt, output_dir, transcribe_args
        ))

    print(f"Generating subtitles for {len(transcribe_tasks)} videos using {max_workers} processes...")

    # Use ProcessPoolExecutor for CPU-intensive transcription
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_path = {executor.submit(transcribe_single_audio, task): task[1] for task in transcribe_tasks}

        # Process completed tasks with progress bar
        with tqdm(total=len(transcribe_tasks), desc="Subtitle generation") as pbar:
            for future in as_completed(future_to_path):
                video_path, srt_path = future.result()
                if srt_path:
                    subtitles_path[video_path] = srt_path
                pbar.update(1)

    return subtitles_path


def get_subtitles(audio_paths: dict, output_srt: bool, output_dir: str, transcribe: callable):
    """Legacy function for backward compatibility."""
    subtitles_path = {}

    for path, audio_path in audio_paths.items():
        srt_path = output_dir if output_srt else tempfile.gettempdir()
        srt_path = os.path.join(srt_path, f"{filename(path)}.srt")

        print(
            f"Generating subtitles for {filename(path)}... This might take a while."
        )

        warnings.filterwarnings("ignore")
        result = transcribe(audio_path)
        warnings.filterwarnings("default")

        with open(srt_path, "w", encoding="utf-8") as srt:
            write_srt(result["segments"], file=srt)

        subtitles_path[path] = srt_path

    return subtitles_path


def encode_single_video(args_tuple):
    """Encode a single video with subtitles. Used for multiprocessing."""
    video_path, srt_path, output_dir = args_tuple

    try:
        out_path = os.path.join(output_dir, f"{filename(video_path)}.mp4")

        video = ffmpeg.input(video_path)
        audio = video.audio

        ffmpeg.concat(
            video.filter('subtitles', srt_path, force_style="OutlineColour=&H40000000,BorderStyle=3"),
            audio, v=1, a=1
        ).output(out_path).run(quiet=True, overwrite_output=True)

        return video_path, out_path
    except Exception as e:
        print(f"Error encoding video {filename(video_path)}: {e}")
        return video_path, None


def encode_videos_parallel(subtitles_paths: dict, output_dir: str, max_threads: int = 2):
    """Encode multiple videos with subtitles in parallel."""

    # Prepare encoding tasks
    encoding_tasks = []
    for video_path, srt_path in subtitles_paths.items():
        encoding_tasks.append((video_path, srt_path, output_dir))

    print(f"Encoding {len(encoding_tasks)} videos with subtitles using {max_threads} threads...")

    results = {}
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        # Submit all tasks
        future_to_path = {executor.submit(encode_single_video, task): task[0] for task in encoding_tasks}

        # Process completed tasks with progress bar
        with tqdm(total=len(encoding_tasks), desc="Video encoding") as pbar:
            for future in as_completed(future_to_path):
                video_path, out_path = future.result()
                if out_path:
                    results[video_path] = out_path
                    print(f"Saved subtitled video to {os.path.abspath(out_path)}")
                pbar.update(1)

    return results


def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def print_memory_usage(stage):
    """Print current memory usage for a given stage."""
    memory_mb = get_memory_usage()
    print(f"Memory usage after {stage}: {memory_mb:.1f} MB")


def optimize_workers_for_memory(video_count, model_name, max_workers):
    """Optimize number of workers based on available memory and model size."""
    available_memory_gb = psutil.virtual_memory().available / (1024**3)

    # Estimate memory usage per worker based on model size
    model_memory_estimates = {
        'tiny': 0.5, 'tiny.en': 0.5,
        'base': 1.0, 'base.en': 1.0,
        'small': 2.0, 'small.en': 2.0,
        'medium': 5.0, 'medium.en': 5.0,
        'large': 10.0
    }

    estimated_memory_per_worker = model_memory_estimates.get(model_name, 5.0)
    safe_workers = max(1, int(available_memory_gb * 0.8 / estimated_memory_per_worker))

    optimized_workers = min(max_workers, safe_workers, video_count)

    if optimized_workers < max_workers:
        print(f"Reducing workers from {max_workers} to {optimized_workers} due to memory constraints")
        print(f"Available memory: {available_memory_gb:.1f} GB, estimated per worker: {estimated_memory_per_worker:.1f} GB")

    return optimized_workers


def process_videos_parallel(video_paths, model_name, output_dir, output_srt, srt_only,
                          max_workers, threads_per_worker, transcribe_args):
    """Main function to process multiple videos in parallel."""

    print(f"Processing {len(video_paths)} videos...")
    print_memory_usage("startup")

    # Optimize workers based on available memory
    optimized_workers = optimize_workers_for_memory(len(video_paths), model_name, max_workers)

    print(f"Configuration: {optimized_workers} workers, {threads_per_worker} threads per worker")

    # Step 1: Extract audio in parallel
    print("\n=== Step 1: Audio Extraction ===")
    audio_paths = get_audio_parallel(video_paths, max_threads=threads_per_worker * optimized_workers)
    print_memory_usage("audio extraction")

    if not audio_paths:
        print("No audio files were successfully extracted.")
        return

    # Step 2: Generate subtitles in parallel
    print("\n=== Step 2: Subtitle Generation ===")
    subtitles_paths = get_subtitles_parallel(
        audio_paths, output_srt or srt_only, output_dir,
        model_name, transcribe_args, max_workers=optimized_workers
    )
    print_memory_usage("subtitle generation")

    if not subtitles_paths:
        print("No subtitles were successfully generated.")
        return

    # Step 3: If srt_only, we're done
    if srt_only:
        print(f"Generated {len(subtitles_paths)} SRT files.")
        return

    # Step 4: Encode videos with subtitles in parallel
    print("\n=== Step 3: Video Encoding ===")
    encoded_videos = encode_videos_parallel(
        subtitles_paths, output_dir, max_threads=threads_per_worker * optimized_workers
    )
    print_memory_usage("video encoding")

    print(f"\n✅ Successfully processed {len(encoded_videos)} videos with subtitles.")


if __name__ == '__main__':
    main()
