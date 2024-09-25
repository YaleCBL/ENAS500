from music21 import note, stream, midi
from midi2audio import FluidSynth

# Create a function to handle note and rest creation
def add_notes_to_theme(theme, pitches, durations):
    for pitch, duration in zip(pitches, durations):
        if pitch == "REST":
            n = note.Rest()  # Handle rests
        else:
            n = note.Note(pitch)  # Create note
        n.quarterLength = duration  # Set duration
        theme.append(n)

# Define the pitches and durations for the main theme (as in the initial code)
pitches1 = ["E5", "D#5", "E5", "D#5", "E5", "B4", "D5", "C5"]
durations1 = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
pitches2 = ["A4", "REST", "C4", "E4", "A4", "B4", "REST", "E4"]
durations2 = [2.0, 0.5, 0.5, 0.5, 0.5, 2.0, 0.5, 0.5]
pitches3 = ["G#4", "B4", "C5", "REST", "E4"]
durations3 = [0.5, 0.5, 2.0, 0.5, 0.5]
pitches4 = ["C5", "B4", "A4"]
durations4 = [0.5, 0.5, 2.0]

# Define the next part of the piece
pitches5 = ["E4", "G#4", "B4", "C5", "E4", "G#4", "B4", "C5", "REST"]
durations5 = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 2.0]
pitches6 = ["C5", "E4", "G#4", "A4", "C5", "E4", "A4"]
durations6 = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 2.0]

# Define the return to the original theme
pitches7 = ["E5", "D#5", "E5", "D#5", "E5", "B4", "D5", "C5"]
durations7 = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

# Create a stream (like the Phrase object)
theme = stream.Stream()

# Add notes to the stream for each section
add_notes_to_theme(theme, pitches1, durations1)  # Main theme
add_notes_to_theme(theme, pitches2, durations2)  # Second part
add_notes_to_theme(theme, pitches3, durations3)  # Transition
add_notes_to_theme(theme, pitches1, durations1)  # Repeat
add_notes_to_theme(theme, pitches2, durations2)  # Second part repeat
add_notes_to_theme(theme, pitches4, durations4)  # Ending part of theme
add_notes_to_theme(theme, pitches5, durations5)  # New section
add_notes_to_theme(theme, pitches6, durations6)  # New section
add_notes_to_theme(theme, pitches7, durations7)  # Return to the original theme

# Save the extended theme as a MIDI file
mf = midi.translate.music21ObjectToMidiFile(theme)
mf.open("fur_elise_extended.mid", 'wb')
mf.write()
mf.close()
