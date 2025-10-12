import os

input_file = "europarl-v7.fr-en.fr"  # your original file
split_line = 1600000     # number of lines for the train file

# Derive base name and extension
base_name, ext = os.path.splitext(input_file)
train_file = f"{base_name}_train{ext}"
eval_file = f"{base_name}_eval{ext}"

with open(input_file, 'r', encoding='utf-8') as infile, \
     open(train_file, 'w', encoding='utf-8') as train_out, \
     open(eval_file, 'w', encoding='utf-8') as eval_out:

    for current_line, line in enumerate(infile, start=1):
        if current_line <= split_line:
            train_out.write(line)
        else:
            eval_out.write(line)

print(f"Done! First {split_line} lines -> {train_file}, remaining lines -> {eval_file}")
