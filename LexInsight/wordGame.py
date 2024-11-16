import pygame
import random
import time
import cv2
from gaze_tracking import GazeTracking

# Initialize pygame
pygame.init()

# Initialize GazeTracking
gaze = GazeTracking()
webcam = cv2.VideoCapture(0)

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
gaze_times = {"left": 0, "center": 0, "right": 0}  # Tracks gaze time for sections
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
round_start_time = time.time()

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
            (5.8 * WIDTH // 6 - 150, HEIGHT // 2)  # Right  ''' Fix this later '''
        ]

        for word, (x, y) in zip(words_for_current_round, word_positions):
            display_text_at_position(word, x, y)

        # Track the time spent looking at each section
        if is_left:
            gaze_times["left"] += 1
        elif is_center:
            gaze_times["center"] += 1
        elif is_right:
            gaze_times["right"] += 1

        pygame.display.flip()

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    # Record the word looked at the longest for the current round
                    max_section = max(gaze_times, key=gaze_times.get)
                    max_time = gaze_times[max_section]
                    word_looked_at = words_for_current_round[["left", "center", "right"].index(max_section)]

                    round_results.append((word_looked_at, max_time))

                    # Display the result of the round
                    screen.fill(BLACK)
                    display_text_at_position(f"Round {current_round} Results", WIDTH // 2 - 150, HEIGHT // 2 - 100)
                    display_text_at_position(f"Word: {word_looked_at}", WIDTH // 2 - 150, HEIGHT // 2)
                    display_text_at_position(f"Time: {max_time / 30:.2f} seconds", WIDTH // 2 - 150, HEIGHT // 2 + 50)
                    pygame.display.flip()
                    time.sleep(2)

                    # Move to the next round
                    current_round += 1
                    if current_round <= rounds:
                        words_for_current_round = random.sample(word_list, 3)  # Update words for the new round
                        gaze_times = {"left": 0, "center": 0, "right": 0}  # Reset gaze times
                    round_start_time = time.time()
                elif event.key == pygame.K_q:
                    # Quit the game when 'q' is pressed
                    running = False

    else:
        # End screen
        screen.fill(BLACK)
        display_text_at_position("Game Over!", WIDTH // 2 - 100, HEIGHT // 2 - 100)

        # Display results for all rounds
        y_offset = -50
        for i, (word, time_spent) in enumerate(round_results, 1):
            display_text_at_position(f"Round {i}: {word} - {time_spent / 30:.2f}s", WIDTH // 2 - 200, HEIGHT // 2 + y_offset, small=True)
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

# Print the results
print("Results for Each Round:")
for i, (word, time_spent) in enumerate(round_results, 1):
    print(f"Round {i}: {word} - {time_spent / 30:.2f} seconds")
