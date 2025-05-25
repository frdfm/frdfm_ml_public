import random, csv

a = ['cat', 'dog', 'pig', 'bat', 'rat', 'fox', 'cow', 'ant', 'elk', 'hen']
with open('data.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['en', 'fr'])
    for _ in range(1000_000):
        s = [random.choice(a) for _ in range(random.randint(2, 4))]
        w.writerow([' '.join(s), ' '.join(sorted(set(s)))])
        # w.writerow([' '.join(s), ' '.join(s)])
