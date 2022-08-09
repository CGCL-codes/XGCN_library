import sys

def run(infile, outfile):
    user2items = {}
    with open(infile, 'r') as rd:
        while True:
            line = rd.readline()
            if not line:
                break
            words = line[:-1].split(' ')
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
    run(infile, outfile)