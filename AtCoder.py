def practiceA():
    _a = int(input())
    _bc = input().split()
    _b = int(_bc[0])
    _c = int(_bc[1])
    _s = str(input())
    print("{} {}".format(_a + _b + _c, _s))


def agc021A():
    # 1~inputまでの数字で最大の桁和を出力
    _num = int(input())
    _num2 = _num
    _digit = 0
    # _numの桁数を求める
    while _num2:
        _digit += 1
        _num2 //= 10
    _topdig = _num // 10 ** (_digit - 1)
    # _numが X999...999の場合は_numが桁和最大
    if (_num - _topdig * 10 ** (_digit - 1)) == 10 ** (_digit - 1) - 1:
        print(_topdig + 9 * (_digit - 1))
    # それ以外は(X-1)999...999が桁和最大
    else:
        print(_topdig - 1 + 9 * (_digit - 1))


def agc022A():
    from collections import OrderedDict
    S = input()
    alfa_list = [chr(i) for i in range(97, 97 + 26)]
    alfa_dict = OrderedDict()

    # {"a":0, ...}
    for i in range(26):
        alfa_dict[alfa_list[i]] = i

    S_max = "zyxwvutsrqponmlkjihgfedcba"
    S_min = "abcdefghijklmnopqrstuvwxyz"

    if S == S_max:
        print(-1)

    elif len(S) < 26:
        for i in range(len(S)):
            del alfa_dict[S[i]]

        remainings = [k for k, v in alfa_dict.items()]
        out = S + remainings[0]
        print(out)
        return 0

    else:
        alfa_dict2 = alfa_dict.copy()
        for i in range(26):
            tmp = alfa_dict[S[i]]
            del alfa_dict[S[i]]

            if S[i] != S_min[i]:
                remainings = [k for k, v in alfa_dict.items() if v > alfa_dict2[S[i - 1]]]

                if S[i:] == S_max[:26 - i]:
                    out = S[:i - 1] + remainings[0]
                    print(out)
                    return 0
                else:
                    out = S[:i + 1] + remainings[1]
                    print(out)
                    return 0

        print("abcdefghijklmnopqrstuvwxz")


def arc094C():
    abc = input().split()
    abc = list(map(int, abc))
    abc.sort()

    count = abc[2] - abc[1]
    diff = abc[2] - (abc[0] + count)
    count += [diff // 2 + 2, diff // 2][diff % 2 == 0]

    print(count)


def abc049C():
    s = input()
    s = s.replace("eraser", "")
    s = s.replace("erase", "")
    s = s.replace("dreamer", "")
    s = s.replace("dream", "")
    # この順番に削除すること
    print(["NO", "YES"][s == ""])


def abc081A():
    s = map(int, input())
    print(sum(s))


def abc081B():
    n = int(input())
    a = input().split()
    a = [int(x) for x in a]
    counts = []

    for num in a:
        tmp = 0
        while num % 2 == 0:
            tmp += 1
            num = num / 2
        if tmp == 0:
            print(0)
            return 0
        else:
            counts.append(tmp)
    print(min(counts))


def abc083B():
    s = input().split()
    n = int(s[0])
    a = int(s[1])
    b = int(s[2])
    sum = 0
    for i in range(1, n + 1):
        tmp = i
        digsum = 0
        while i:
            digsum += i % 10
            i //= 10
        if a <= digsum <= b:
            sum += tmp
    print(sum)


def abc085B():
    n = int(input())
    d = []
    for i in range(n):
        d.append(int(input()))
    d = list(set(d))
    print(len(d))


def abc085C():
    s = input().split()
    n = int(s[0])
    y = int(s[1])

    for i in range(y // 10000 + 1):
        for j in range((y - 10000 * i) // 5000 + 1):
            if 10000 * i + 5000 * j + 1000 * (n - i - j) == y:
                print(i, j, n - i - j)
                return 0
    print(-1, -1, -1)


def abc086A():
    s = input().split()
    a = int(s[0])
    b = int(s[1])
    print(["Even", "Odd"][a * b % 2])


def abc086C():
    import numpy as np
    n = int(input())
    txys = np.empty((n, 3))
    for i in range(n):
        s = input().split()
        txys[i] = (list(map(int, s)))
    cur = np.array([0, 0, 0])
    for ele in txys:
        tmp = ele - cur
        if tmp[0] >= abs(tmp[1]) + abs(tmp[2]) \
                and (tmp[0] - abs(tmp[1]) - abs(tmp[2])) % 2 == 0:
            cur = ele
        else:
            print("No")
            return 0
    print("Yes")


def abc087B():
    a = int(input())
    b = int(input())
    c = int(input())
    x = int(input())
    count = 0
    for i in range(a + 1):
        for j in range(b + 1):
            for k in range(c + 1):
                if 500 * i + 100 * j + 50 * k == x:
                    count += 1
    print(count)


def abc088B():
    n = int(input())
    a = input().split()
    a = [int(i) for i in a]
    a.sort()
    a = a[::-1]
    alice = a[::2]
    bob = a[1::2]
    print(sum(alice) - sum(bob))


def abc089A():
    _num = int(input())
    print(_num // 3)


def abc089B():
    _num = int(input())
    del _num
    _colors = str(input())
    if _colors.find("Y") == -1:
        print("Three")
    else:
        print("Four")


def abc089C():
    _num = int(input())
    _namelist = []
    _countM = 0
    _countA = 0
    _countR = 0
    _countC = 0
    _countH = 0

    for i in range(0, _num):
        _name = str(input())
        _namelist.append(_name)
        if _name[0] == "M":
            _countM += 1
        elif _name[0] == "A":
            _countA += 1
        elif _name[0] == "R":
            _countR += 1
        elif _name[0] == "C":
            _countC += 1
        elif _name[0] == "H":
            _countH += 1
    print(_countM * _countA * _countR + \
          _countM * _countA * _countC + \
          _countM * _countA * _countH + \
          _countM * _countR * _countC + \
          _countM * _countR * _countH + \
          _countM * _countC * _countH + \
          _countA * _countR * _countC + \
          _countA * _countR * _countH + \
          _countA * _countC * _countH + \
          _countR * _countC * _countH)


def abc089D():
    _hwd = input().split()
    _hwd = list(map(int, _hwd))
    _area = []
    _lrs = []
    _nowx = 0
    _nowy = 0

    for i in range(0, _hwd[0]):
        _aij = str(input()).split()
        _area.append(list(map(int, _aij)))

    _q = int(input())

    for i in range(0, _q):
        _lr = str(input()).split()
        _lrs.append(list(map(int, _lr)))

    for i in range(0, _q):
        _mpcount = 0
        for i2 in range(0, int((_lrs[i][1] - _lrs[i][0]) / _hwd[2]) + 1):
            for i3 in range(0, _hwd[0]):
                if _lrs[i][0] + _hwd[2] * i2 in _area[i3]:
                    if not i2:
                        _nowx = _area[i3].index(_lrs[i][0]) + 1
                        _nowy = i3 + 1
                    _nextx = _area[i3].index(_lrs[i][0] + _hwd[2] * i2) + 1
                    _nexty = i3 + 1
            _mpcount += abs(_nextx - _nowx) + abs(_nexty - _nowy)
            _nowx = _nextx
            _nowy = _nexty
        print(_mpcount)


def abc091A():
    _instr = str(input()).split()
    _a = int(_instr[0])
    _b = int(_instr[1])
    _c = int(_instr[2])
    if _a + _b >= _c:
        print("Yes")
    else:
        print("No")


def abc091B():
    from collections import Counter
    _strs = []
    _strt = []
    _max = 0
    _n = int(input())
    for i in range(_n):
        _strs.append(str(input()))
    _m = int(input())
    for i in range(_m):
        _strt.append(str(input()))
    _strs2 = list(set(_strs))
    for element in _strs2:
        _score = Counter(_strs)[element] - Counter(_strt)[element]
        if _score > _max:
            _max = _score
    print(_max)


def abc091C():
    from operator import itemgetter
    _reds = []
    _blues = []
    _count = 0

    _n = int(input())

    for i in range(_n):
        _instr = str(input()).split()
        _reds.append(list(map(int, _instr)))

    for i in range(_n):
        _instr = str(input()).split()
        _blues.append(list(map(int, _instr)))

    _reds.sort()
    _blues.sort(key=itemgetter(1))

    for i in range(len(_reds)):
        _num = 0
        _blue2 = []
        while _reds[i][1] < _blues[_num][1]:
            _blue2.append(_blues[_num])
            _num += 1
        _blue2.sort()
        if len(_blue2) > 0:
            if _reds[i][0] < _blue2[0][0]:
                _count += 1
                _blues.remove(_blue2[0])
        """for j in range(len(_blues)):
            if _reds[i][0] >= _blues[j][0]:
                continue
            elif _reds[i][1] < _blues[j][1]:
                _count += 1
                del _blues[j]
                break
            else:
                continue"""
    print(_count)
