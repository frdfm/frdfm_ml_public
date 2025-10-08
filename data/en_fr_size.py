with open('en-fr.csv', 'r', encoding='utf-8') as f:
    count = sum(1 for _ in f)
print(count)
