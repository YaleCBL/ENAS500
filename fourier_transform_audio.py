import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.io.wavfile as wavfile
import subprocess
import sys
import os

def main():
    # Check if fluidsynth is installed
    try:
        subprocess.run(['fluidsynth', '-h'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        print("Error: fluidsynth is not installed or not in PATH.")
        sys.exit(1)

    # Check if FFmpeg is installed
    try:
        subprocess.run(['ffmpeg', '-h'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        print("Error: FFmpeg is not installed or not in PATH.")
        sys.exit(1)

    # Get the MIDI file from command line argument
    if len(sys.argv) < 2:
        print("Usage: python script.py input.mid")
        sys.exit(1)
    midi_file = sys.argv[1]

    # Check if the MIDI file exists
    if not os.path.exists(midi_file):
        print(f"Error: MIDI file '{midi_file}' not found.")
        sys.exit(1)

    # Define the output WAV file
    wav_file = 'output.wav'

    # Define the soundfont file
    soundfont_file = 'soundfont.sf2'
    if not os.path.exists(soundfont_file):
        print(f"Error: Soundfont file '{soundfont_file}' not found.")
        print("Please download a soundfont file and place it in the current directory as 'soundfont.sf2'.")
        sys.exit(1)

    # Render the MIDI file to WAV using fluidsynth
    print("Rendering MIDI to WAV...")
    subprocess.run(['fluidsynth', '-ni', soundfont_file, midi_file, '-F', wav_file, '-r', '44100'])

    # Read in the WAV file
    print("Reading WAV file...")
    sample_rate, audio_data = wavfile.read(wav_file)

    # If stereo, take one channel
    if len(audio_data.shape) == 2:
        audio_data = audio_data[:, 0]

    # Normalize audio data to [-1, 1]
    audio_data = audio_data / np.max(np.abs(audio_data))

    # Increase font sizes
    plt.rcParams.update({'font.size': 16})

    # Set up the animation
    fig, (ax_time, ax_freq) = plt.subplots(2, 1, figsize=(19.2, 10.8), constrained_layout=True)
    fig.patch.set_facecolor('black')

    # Set up axes
    ax_time.set_facecolor('black')
    ax_freq.set_facecolor('black')

    for ax in [ax_time, ax_freq]:
        ax.tick_params(axis='x', colors='white', labelsize=14)
        ax.tick_params(axis='y', colors='white', labelsize=14)
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')

    ax_time.set_title('Audio Signal (Time Domain)', fontsize=20)
    ax_time.set_xlabel('Time [s]', fontsize=18)
    ax_time.set_ylabel('Amplitude', fontsize=18)

    ax_freq.set_title('Fourier Transform (Frequency Domain)', fontsize=20)
    ax_freq.set_xlabel('Frequency [Hz]', fontsize=18)
    ax_freq.set_ylabel('Magnitude', fontsize=18)

    # Initialize the plot lines
    line_time, = ax_time.plot([], [], color='red')
    line_freq, = ax_freq.plot([], [], color='red')

    # Set axis limits
    window_size = 8192  # Increased number of samples per frame for higher frequency resolution
    total_samples = len(audio_data)
    duration = total_samples / sample_rate

    fps = 30  # Desired frames per second
    num_frames = int(duration * fps)

    # Recalculate hop_size to synchronize with the desired FPS
    hop_size = int(sample_rate / fps)

    time_axis = np.linspace(0, window_size / sample_rate, num=window_size)
    freq_axis = np.linspace(0, sample_rate / 2, num=window_size // 2 + 1)

    # Truncate the frequency domain at 1000 Hz
    max_freq = 1000  # Maximum frequency to display
    freq_limit_idx = np.searchsorted(freq_axis, max_freq, side='right')

    ax_time.set_xlim(0, window_size / sample_rate)
    ax_time.set_ylim(-1, 1)

    ax_freq.set_xlim(0, max_freq)
    ax_freq.set_ylim(0, 1)

    def animate(i):
        start = i * hop_size
        end = start + window_size

        # Ensure we don't go beyond the audio data
        if end > total_samples:
            end = total_samples
            start = end - window_size
            if start < 0:
                start = 0

        window_data = audio_data[start:end]

        # If window_data is shorter than window_size, pad with zeros
        if len(window_data) < window_size:
            window_data = np.pad(window_data, (0, window_size - len(window_data)), 'constant')

        # Update time domain plot
        line_time.set_data(time_axis, window_data)

        # Compute Fourier Transform
        fft_data = np.abs(np.fft.rfft(window_data))
        fft_data /= np.max(fft_data)  # Normalize

        # Truncate frequency data to max_freq
        fft_data_truncated = fft_data[:freq_limit_idx]
        freq_axis_truncated = freq_axis[:freq_limit_idx]

        line_freq.set_data(freq_axis_truncated, fft_data_truncated)

        return line_time, line_freq

    anim = animation.FuncAnimation(
        fig, animate, frames=num_frames, interval=1000 / fps, blit=True)

    # Save the animation
    print("Saving animation...")
    video_file = 'output.mp4'

    # Use FFMpegWriter
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=fps, metadata=dict(artist='Python Script'), bitrate=1800)

    # Save at 1920x1080 resolution
    anim.save(video_file, writer=writer, dpi=100)

    # Combine video and audio using ffmpeg with increased audio volume
    print("Combining video and audio with increased volume...")
    final_video = 'final_output.mp4'
    subprocess.run([
        'ffmpeg', '-y', '-i', video_file, '-i', wav_file,
        '-filter:a', 'volume=3.0',  # Increase volume by a factor of 3
        '-c:v', 'copy', '-c:a', 'aac', '-strict', 'experimental', final_video
    ])

    print("Done. Output video saved as 'final_output.mp4'.")

if __name__ == '__main__':
    main()
