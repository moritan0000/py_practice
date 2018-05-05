def practice_a():
    a = int(input())
    b, c = map(int, input().split())
    s = input()
    print(a + b + c, s)


def agc021_a():
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


def agc022_a():
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


def arc094_c():
    abc = list(map(int, input().split()))
    abc.sort()

    count = abc[2] - abc[1]
    diff = abc[2] - (abc[0] + count)
    count += [diff // 2 + 2, diff // 2][diff % 2 == 0]

    print(count)


def abc049_c():
    s = input()
    s = s.replace("eraser", "")
    s = s.replace("erase", "")
    s = s.replace("dreamer", "")
    s = s.replace("dream", "")
    # この順番に削除すること
    print(["NO", "YES"][s == ""])


def abc081_a():
    s = map(int, input())
    print(sum(s))


def abc081_b():
    n = int(input())
    a = list(map(int, input().split()))
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


def abc083_b():
    n, a, b = map(int, input().split())
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


def abc085_b():
    n = int(input())
    d = []
    for i in range(n):
        d.append(int(input()))
    d = list(set(d))
    print(len(d))


def abc085_c():
    n, y = map(int, input().split())

    for i in range(y // 10000 + 1):
        for j in range((y - 10000 * i) // 5000 + 1):
            if 10000 * i + 5000 * j + 1000 * (n - i - j) == y:
                print(i, j, n - i - j)
                return 0
    print(-1, -1, -1)


def abc086_a():
    a, b = map(int, input().split())
    print(["Even", "Odd"][a * b % 2])


def abc086_c():
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


def abc087_b():
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


def abc088_b():
    n = int(input())
    a = input().split()
    a = [int(i) for i in a]
    a.sort()
    a = a[::-1]
    alice = a[::2]
    bob = a[1::2]
    print(sum(alice) - sum(bob))


def abc089_a():
    print(int(input()) // 3)


def abc089_b():
    num = int(input())
    colors = input()
    print(["Four", "Three"][colors.find("Y") == -1])


def abc089_c():
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
    print(_countM * _countA * _countR +
          _countM * _countA * _countC +
          _countM * _countA * _countH +
          _countM * _countR * _countC +
          _countM * _countR * _countH +
          _countM * _countC * _countH +
          _countA * _countR * _countC +
          _countA * _countR * _countH +
          _countA * _countC * _countH +
          _countR * _countC * _countH)


def abc089_d():
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


def abc091_a():
    a, b, c = map(int, input().split())
    print(["No", "Yes"][a + b >= c])


def abc091_b():
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


def abc091_c():
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


def abc094_a():
    a, b, x = map(int, input().split())
    print(["NO", "YES"][a <= x <= (a + b)])


def abc094_b():
    n = list(map(int, input().split()))
    a = list(map(int, input().split()))
    count = 0
    for i in a:
        if n[2] > i:
            count += 1
    print(min(count, n[1] - count))


def abc094_c():
    n = int(input())
    x = list(map(int, input().split()))
    y = x.copy()
    y.sort()
    nums = y[int(n / 2 - 1): int(n / 2 + 1)]
    for i in range(n):
        tf = nums[0] >= x[i]
        print(nums[tf])


def abc094_d():
    import numpy as np

    def ncr(n, r):
        """
        Calculate the number of combination (nCr = nPr/r!).
        The parameters need to meet the condition of n >= r >= 0.
        It returns 1 if r == 0, which means there is one pattern
        to choice 0 items out of the number of n.
        """
        r = min(r, n - r)
        numerator = 1
        denominator = 1
        for i in range(r):
            numerator *= (n - i)
            denominator *= (i + 1)

        return numerator // denominator

    n = int(input())
    a = list(map(float, input().split()))
    a.sort()

    if len(a) == 2:
        print(a[1], a[0])
        return 0

    a = np.array(a)
    ai = int(a[-1])

    if n % 2 == 0:
        r = ai / 2
        b = a.copy() - r
        j = np.argmin(abs(b))
        print(int(ai), int(a[j]))
        return 0
    else:
        r1 = (ai - 1) / 2
        r2 = (ai + 1) / 2
        b1 = a.copy() - r1
        b2 = a.copy() - r2

        j1 = np.argmin(abs(b1))
        j2 = np.argmin(abs(b2))
        # j = np.argmin([abs(a[j1] - r1), abs(r2 - a[j2])])
        if abs(a[j1] - r1) == abs(a[j2] - r2):
            j = np.argmax([ncr(ai, int(a[j1])), ncr(ai, int(a[j2]))])
        else:
            j = np.argmin([abs(a[j1] - r1), abs(r2 - a[j2])])
        aj = [a[j1], a[j2]][j[0]]
        print(int(ai), int(aj))
        return 0


def abc095_a():
    s = input()
    print(700 + 100 * s.count("o"))


def abc095_b():
    n, x = map(int, input().split())
    m = []
    for i in range(n):
        m.append(int(input()))

    x -= sum(m)
    count = x // min(m)
    print(n + count)


def abc095_c():
    a, b, c, x, y = map(int, input().split())

    diff = abs(x - y)
    price1 = [(a + b) * min(x, y), 2 * c * min(x, y)][a + b >= c * 2] + diff * [a, b][x < y]
    print(min(price1, max(x, y) * c * 2))


def abc095_dx():
    import numpy as np
    n, c = map(int, input().split())
    x = np.zeros((n, 1))
    v = np.zeros((n, 1))

    for i in range(n):
        x[i], v[i] = map(int, input().split())

    for i in range(n):
        xr = x[i:]
        xr = [min(xr[i] - xr[0], xr[0] + c - xr[i]) for i in range(n)]
        vr = v[i:]
        xl = x[:i]
        vl = v[:i]

        print(xr)


def abc096_a():
    pass


print(abc096_a())


def abc096_b():
    pass


print(abc096_b())


def abc096_c():
    pass


print(abc096_c())


def abc096_d():
    pass


print(abc096_d())
