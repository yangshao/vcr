import re
def process(sent):
    pattern = 'DB[1-9]+'
    l = re.findall(pattern, sent)
    new_sent  = re.sub(pattern, "", sent)
    l.append(new_sent)
    return l
if __name__ == '__main__':
    sent = 'I hate DB124 and DB568 .'
    res = process(sent)
    print(sent)
    print(res)