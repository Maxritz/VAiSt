t = open('README.md', encoding='utf-8').read()
blocks = t.split('```')
tree = blocks[1]
print('vaist/ in tree:', 'vaist/' in tree)
print('lines with vaist:')
for l in tree.split('\n'):
    if 'vaist' in l.lower():
        print('  ', l)
