import pygame
import random
import time
import cv2
import threading
from gaze_tracking import GazeTracking
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from scipy.signal import butter, filtfilt
import numpy as np
import matplotlib.pyplot as plt

# Initialize pygame
pygame.init()

# Initialize GazeTracking
gaze = GazeTracking()
webcam = cv2.VideoCapture(0)

# Global variables for cognitive spike detection and EEG data
eeg_data_collected = []
deviation_records = []  # Records deviations from the mean
deviation_threshold = 50  # Hardcoded standard deviation threshold
mean_eeg = 0
muse_connected = False  # Flag to indicate if Muse 2 is connected

def butter_bandpass(lowcut, highcut, fs, order=5):
    """Create a bandpass filter."""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    """Apply a bandpass filter to the data."""
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return filtfilt(b, a, data)

def plot_eeg_data(raw_data, filtered_data, time_axis, deviation_records, channel_name):
    """Plot raw vs filtered EEG signals with deviation markers."""
    plt.figure(figsize=(10, 6))
    plt.plot(time_axis, raw_data[0, :], label=f"Raw Signal ({channel_name})", alpha=0.6)
    plt.plot(time_axis, filtered_data[0, :], label=f"Filtered Signal ({channel_name})", alpha=0.9)

    # Highlight deviations
    for record in deviation_records:
        plt.axvspan(record["start_time"], record["end_time"], color="red", alpha=0.3, label="Deviation")

    plt.title(f"Raw vs Filtered EEG Signal ({channel_name})", fontsize=12)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (μV)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

def monitor_muse_activity():
    """Monitor Muse 2 data and track deviations from the mean EEG signal."""
    global eeg_data_collected, deviation_records, mean_eeg, muse_connected

    params = BrainFlowInputParams()
    params.ip_address = '0.0.0.0'
    params.serial_port = ''

    board_id = BoardIds.MUSE_2_BOARD.value
    board = BoardShim(board_id, params)

    try:
        print("Attempting to connect to Muse 2...")
        board.prepare_session()
        print("Muse 2 connected successfully!")
        muse_connected = True  # Indicate that the Muse is connected
        board.start_stream()

        fs = BoardShim.get_sampling_rate(board_id)  # Sampling rate
        eeg_channels = BoardShim.get_eeg_channels(board_id)
        left_hemisphere_channel = [1]  # AF7 channel for Muse 2

        collected_samples = []
        while True:
            # Fetch 1 second of data
            data = board.get_current_board_data(250)
            eeg_data = data[eeg_channels, :]
            selected_data = eeg_data[left_hemisphere_channel, :]
            collected_samples.append(selected_data)

            # Update mean and track deviations
            if len(collected_samples) > 1:
                combined_data = np.hstack(collected_samples)
                filtered_data = bandpass_filter(combined_data, 0.5, 50, fs)
                mean_eeg = np.mean(filtered_data)

                for i, sample in enumerate(filtered_data[0]):
                    deviation = abs(sample - mean_eeg)
                    if deviation > deviation_threshold:
                        deviation_records.append({
                            "start_time": i / fs,
                            "end_time": (i + 1) / fs,
                            "deviation": deviation
                        })

            eeg_data_collected.append(selected_data)

    except Exception as e:
        print(f"Error in Muse thread: {e}")
    finally:
        board.release_session()

# Start Muse monitoring in a separate thread
muse_thread = threading.Thread(target=monitor_muse_activity, daemon=True)
muse_thread.start()

# Wait until Muse 2 is fully connected
print("Waiting for Muse 2 to connect...")
while not muse_connected:
    time.sleep(1)

print("Muse 2 is ready. Starting the game...")

# Get the full screen size of the display
info = pygame.display.Info()
WIDTH, HEIGHT = info.current_w, info.current_h

# Set the display to full screen
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)
pygame.display.set_caption("Eye-Tracking Word Game")

# Fonts and Colors
FONT = pygame.font.Font(None, 74)
SMALL_FONT = pygame.font.Font(None, 36)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Load the list of words from random_words.txt
with open("random_words.txt", "r") as file:
    word_list = [line.strip() for line in file]

# Game settings
rounds = 10
current_round = 1
words_for_current_round = []  # Stores words for the current round
round_results = []  # Stores results for each round


def display_text_at_position(text, x, y, small=False):
    """Helper function to display text at a specific position."""
    font = SMALL_FONT if small else FONT
    text_surface = font.render(text, True, WHITE)
    screen.blit(text_surface, (x, y))


# Initialize the first set of random words
if current_round <= rounds:
    words_for_current_round = random.sample(word_list, 3)

running = True

# Game loop
while running:
    # Update the gaze tracker with the webcam feed
    _, frame = webcam.read()
    gaze.refresh(frame)

    # Analyze the user's gaze
    is_left = gaze.is_left()
    is_center = gaze.is_center()
    is_right = gaze.is_right()

    screen.fill(BLACK)

    if current_round <= rounds:
        # Display the round number
        display_text_at_position(f"Round {current_round}", WIDTH // 2 - 100, 50, small=True)

        # Spread the words across the width of the screen
        word_positions = [
            (WIDTH // 6 - 150, HEIGHT // 2),  # Left
            (WIDTH // 2 - 100, HEIGHT // 2),  # Center
            (5.8 * WIDTH // 6 - 150, HEIGHT // 2)  # Right
        ]

        for word, (x, y) in zip(words_for_current_round, word_positions):
            display_text_at_position(word, x, y)

        pygame.display.flip()

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    # Determine the difficult word based on gaze and deviation records
                    if is_left:
                        word_looked_at = words_for_current_round[0]
                    elif is_center:
                        word_looked_at = words_for_current_round[1]
                    elif is_right:
                        word_looked_at = words_for_current_round[2]
                    else:
                        word_looked_at = "None"

                    round_results.append((word_looked_at, len(deviation_records)))

                    # Display the result of the round
                    screen.fill(BLACK)
                    display_text_at_position(f"Round {current_round} Results", WIDTH // 2 - 150, HEIGHT // 2 - 100)
                    display_text_at_position(f"Word: {word_looked_at}", WIDTH // 2 - 150, HEIGHT // 2)
                    display_text_at_position(f"Deviations: {len(deviation_records)}", WIDTH // 2 - 150, HEIGHT // 2 + 50)
                    pygame.display.flip()
                    time.sleep(2)

                    # Move to the next round
                    current_round += 1
                    if current_round <= rounds:
                        words_for_current_round = random.sample(word_list, 3)  # Update words for the new round
                    break
                elif event.key == pygame.K_q:
                    # Quit the game when 'q' is pressed
                    running = False

    else:
        # End screen
        screen.fill(BLACK)
        display_text_at_position("Game Over!", WIDTH // 2 - 100, HEIGHT // 2 - 100)

        # Display results for all rounds
        y_offset = -50
        for i, (word, deviations) in enumerate(round_results, 1):
            display_text_at_position(f"Round {i}: {word} - {deviations} deviations", WIDTH // 2 - 200, HEIGHT // 2 + y_offset, small=True)
            y_offset += 30

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                running = False

pygame.quit()
webcam.release()
cv2.destroyAllWindows()

# Plot EEG data at the end of the game
if eeg_data_collected:
    fs = BoardShim.get_sampling_rate(BoardIds.MUSE_2_BOARD.value)
    combined_data = np.hstack(eeg_data_collected)  # Combine data into a single array
    time_axis = np.linspace(0, combined_data.shape[1] / fs, combined_data.shape[1])
    filtered_data = bandpass_filter(combined_data, 0.5, 50, fs)
    plot_eeg_data(combined_data, filtered_data, time_axis, deviation_records, "Higher-Level Cognitive Function (Left)")
