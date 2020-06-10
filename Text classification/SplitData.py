with open('bbc-text.csv', 'r', encoding='utf-8') as f:
    contents = f.read().splitlines()
    f.close()
del contents[0]
d = {}
for content in contents:
    Y, X = content.split(',')
    if (Y not in d):
        d[Y] = Y
    with open('data/' + d[Y], 'a', encoding='utf-8') as f:
        f.write(X + '\n')
        f.close()

        