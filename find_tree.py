with open('README.md', 'rb') as f:
    data = f.read()
# Find the tree block
start = data.find(b'```\nVAiSt\n')
if start != -1:
    end = data.find(b'```\n', start + 4)
    if end != -1:
        print('Found tree block at', start, 'to', end)
        print('Length:', end - start)
        print('Preview:', data[start:start+200])
    else:
        print('End not found')
else:
    print('Tree block not found')