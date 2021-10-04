import numpy as np

def calMA(new_elem, l, s):
    # 메모리가 너무 찰 수 있음
    l.append(new_elem)
    if len(l) < s:
        return sum(l)/len(l)
    else:
        return sum(l[-s:])/s

def dewel(a1, a2, dewel_time, count):
    a1 = np.asarray(a1)
    a2 = np.asarray(a2)

    a = (a1 * (dewel_time - count) / dewel_time) + (a2 * (count / dewel_time))
    if dewel_time == count:
        f = True
        count = 0
    else:
        f = False
        count += 1
    
    return a, f, count
