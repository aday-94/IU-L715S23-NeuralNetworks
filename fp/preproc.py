""" NNs/Nahuatl Spring 2023 Final Project Andrew Davis -- Preprocessing Get all Source & Target Tokenized Sentences """

""" 
Yes, you can search for the same things in a CoNLL-U file using regular expressions.

CoNLL-U files are tab-separated text files that contain one token per line, with columns representing different annotations like 
the word form, lemma, part-of-speech tag, and syntactic dependency relations. The text and text[orig] attributes are typically not 
included in CoNLL-U files, but you can extract the corresponding word forms using the FORM column.

Here's an example Python code that demonstrates how to extract the sentences that come after text = or text[orig] = in a 
CoNLL-U file: """

#Step 1: Open the conllu file from FC Book and make an out txt file for the token versions of text and text[orig]
import re

filename = "Book_08_-_Kings_and_Lords.conllu"  # Replace with your CoNLL-U file name
OUT = open('toke-tgt-8.txt', 'w')

#Step 2: Define the regular expression pattern to match text after 'text =' or 'text[orig] ='

#pattern = r'(?:text|text\[orig\])\s*=\s*([^\n]+)'
#textog_pattern = r'(?:text\[orig\])\s*=\s*([^\n]+)'
text_pattern = r'(?:text)\s*=\s*([^\n]+)'

#Step 3: Open the file and read its contents

with open(filename, 'r') as file:
    contents = file.read()

#Step 4: Use regular expressions to extract the sentences that match the pattern(s)

sentences = []
for match in re.finditer(text_pattern, contents):
    sentence = match.group(1).strip()
    sentences.append(sentence)

#Step 5: Process the extracted sentences to insert spaces between each char and to create a newline after punctuation

processed_sentences = []
for sentence in sentences:
    sentence = ''.join(sentence.split())  # Remove whitespace
    processed_sentence = ''
    for i, char in enumerate(sentence):
        if i > 0 and char in '.,:;?!':  # Check for punctuation
            processed_sentence += char + '\n'
        else:
            processed_sentence += char + ' '
    processed_sentences.append(processed_sentence)

#Step 6: Print the extracted sentences and store them in a txt file for openNMT

for sentence in processed_sentences:
    print(sentence)
    OUT.write(sentence)

OUT.close()

""" This code first extracts the substrings that match the regular expression pattern using re.finditer(). 
It then iterates over each match and extracts the corresponding word forms using the FORM column. 
The extracted word forms are then joined together to form a sentence.

Note that this code assumes that the CoNLL-U file follows the standard format with the FORM column as the second column. 
If your file has a different format, you may need to modify the code to extract the word forms from the correct column. """