import re

# Define the regular expression pattern
pattern = re.compile(r"([,.;:?!])(?!\n)")

# Open the file for reading and writing
with open('toke-tgt-aug50-1.txt', 'r') as f, open('toke-tgt-aug-1.txt', 'w') as out_f:
    # Loop through each line in the input file
    for line in f:
        # Use the regular expression to find any punctuation marks
        # that don't have a newline after them, and insert a newline
        # after them
        new_line = re.sub(pattern, r"\1\n", line)
        # Write the modified line to the output file
        out_f.write(new_line)
