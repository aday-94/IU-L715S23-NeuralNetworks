""" NNs/Nahuatl Spring 2023 Final Project Andrew Davis -- Data Augmentation Insert Newline Char """


""" This code reads in your source and target files and randomly inserts a newline token in one word per line of text in the source file. 
It then appends the augmented source and target lines to a list and writes the list to a new file called 'augmented.txt'.

Note that the newline token is represented by the '¶' character in this code. You can change this character to something else if 
you prefer. Also note that the code ignores words that are shorter than two characters, so those words won't be augmented. """

import random
import string

with open('toke-tgt-8.txt', 'r') as f_source, open('toke-tgt-aug-8.txt', 'w') as f_augmented:
    for source_line in f_source:
        target_line = f_source.readline().strip()
        source_words = source_line.strip().split()
        for i, word in enumerate(source_words):
            if random.random() < 0.1:  # randomly insert newline token
                source_words[i] = word + " ¶"
        augmented_line = " ".join(source_words) + "\t" + target_line + "\n"
        f_augmented.write(augmented_line)
