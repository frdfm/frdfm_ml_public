file1 = "europarl-v7.fr-en.en"
file2 = "europarl-v7.fr-en.fr"
n = 300  # number of multiples of 10000 you want

targets = {10000 * i for i in range(1, n + 1)}

with open(file1, 'r', encoding='utf-8') as f1, open(file2, 'r', encoding='utf-8') as f2:
    for current_line, (line1, line2) in enumerate(zip(f1, f2), start=1):
        if current_line in targets:
            print(f"Line {current_line}:\nFile1: {line1.rstrip()}\nFile2: {line2.rstrip()}\n")
            targets.remove(current_line)
        if not targets:
            break

last_line_num = 0
last_line_file1 = ""
last_line_file2 = ""

with open(file1, 'r', encoding='utf-8') as f1, open(file2, 'r', encoding='utf-8') as f2:
    for current_line, (line1, line2) in enumerate(zip(f1, f2), start=1):
        last_line_num = current_line
        last_line_file1 = line1.rstrip()
        last_line_file2 = line2.rstrip()

print(f"Last common line ({last_line_num}):\nFile1: {last_line_file1}\nFile2: {last_line_file2}")

# Count lines in file1
with open(file1, 'r', encoding='utf-8') as f1:
    lines_file1 = sum(1 for _ in f1)

# Count lines in file2
with open(file2, 'r', encoding='utf-8') as f2:
    lines_file2 = sum(1 for _ in f2)

print(f"{file1} has {lines_file1} lines")
print(f"{file2} has {lines_file2} lines")