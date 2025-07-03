# Automatic subtitles in your videos

This repository uses `ffmpeg` and [OpenAI's Whisper](https://openai.com/blog/whisper) to automatically generate and overlay subtitles on any video.

## Installation

To get started, you'll need Python 3.7 or newer. Install the binary by running the following command:

    pip install git+https://github.com/sikinder/auto-subtitle.git

You'll also need to install [`ffmpeg`](https://ffmpeg.org/), which is available from most package managers:

```bash
# on Ubuntu or Debian
sudo apt update && sudo apt install ffmpeg

# on MacOS using Homebrew (https://brew.sh/)
brew install ffmpeg

# on Windows using Chocolatey (https://chocolatey.org/)
choco install ffmpeg
```

## Usage

The following command will generate a `subtitled/video.mp4` file contained the input video with overlayed subtitles.

    auto_subtitle /path/to/video.mp4 -o subtitled/

The default setting (which selects the `small` model) works well for transcribing English. You can optionally use a bigger model for better results (especially with other languages). The available models are `tiny`, `tiny.en`, `base`, `base.en`, `small`, `small.en`, `medium`, `medium.en`, `large`.

    auto_subtitle /path/to/video.mp4 --model medium

Adding `--task translate` will translate the subtitles into English:

    auto_subtitle /path/to/video.mp4 --task translate

### 🚀 High-Performance Parallel Processing

Process multiple videos simultaneously with automatic multi-core optimization:

    # Process multiple videos in parallel (auto-detects optimal workers)
    auto_subtitle video1.mp4 video2.mp4 video3.mp4 -o output/

    # Customize parallel processing
    auto_subtitle *.mp4 --max_workers 4 --threads_per_worker 2 -o output/

    # Generate only SRT files (fastest option)
    auto_subtitle *.mp4 --srt_only true -o subtitles/

**Performance improvements:**
- **2-4x faster** processing with parallel execution
- **Automatic memory optimization** based on available RAM
- **Real-time progress tracking** with progress bars
- **Efficient resource usage** with configurable workers and threads

### Performance Options

- `--max_workers`: Number of parallel processes (default: auto-detect)
- `--threads_per_worker`: Threads per worker for I/O operations (default: 2)
- `--srt_only`: Generate only subtitle files, skip video encoding

Run the following to view all available options:

    auto_subtitle --help

## License

This script is open-source and licensed under the MIT License. For more details, check the [LICENSE](LICENSE) file.
