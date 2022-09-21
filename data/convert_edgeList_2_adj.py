import sys

def run(infile, outfile, has_header=False, sep=' '):
    user2items = {}
    with open(infile, 'r') as rd:
        if has_header:
            rd.readline()
        while True:
            line = rd.readline()
            if not line:
                break
            words = line[:-1].split(sep)
            userid, itemid = int(words[0]), int(words[1])
            if userid not in user2items:
                user2items[userid] = []
            user2items[userid].append(itemid)
    with open(outfile, 'w') as wt:
        for k,v in user2items.items():
            wt.write('{0} {1}\n'.format(k, ' '.join([str(a) for a in v])))


if __name__ == '__main__':
    infile = sys.argv[1]
    outfile = sys.argv[2]
    
    has_header = False
    if len(sys.argv) >= 4:
        has_header = bool(int(sys.argv[3]))
    
    sep = ' '
    if len(sys.argv) >= 5:
        sep = sys.argv[4]
    run(infile, outfile, has_header, sep)