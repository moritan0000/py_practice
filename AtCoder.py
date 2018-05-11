def practice_a():
    a = int(input())
    b, c = map(int, input().split())
    s = input()
    return a + b + c, s


def agc021_a():
    # 1~inputまでの数字で最大の桁和を出力
    n = int(input())
    n_2 = n
    _digit = 0
    # _numの桁数を求める
    while n_2:
        _digit += 1
        n_2 //= 10
    top_dig = n // 10 ** (_digit - 1)
    # _numが X999...999の場合は_numが桁和最大
    if (n - top_dig * 10 ** (_digit - 1)) == 10 ** (_digit - 1) - 1:
        print(top_dig + 9 * (_digit - 1))
    # それ以外は(X-1)999...999が桁和最大
    else:
        print(top_dig - 1 + 9 * (_digit - 1))


def agc022_a():
    from collections import OrderedDict
    s = input()
    alpha_list = [chr(i) for i in range(97, 97 + 26)]
    alpha_dict = OrderedDict()

    # {"a":0, ...}
    for i in range(26):
        alpha_dict[alpha_list[i]] = i

    s_max = "zyxwvutsrqponmlkjihgfedcba"
    s_min = "abcdefghijklmnopqrstuvwxyz"

    if s == s_max:
        print(-1)

    elif len(s) < 26:
        for i in range(len(s)):
            del alpha_dict[s[i]]

        remaining = [k for k, v in alpha_dict.items()]
        out = s + remaining[0]
        print(out)
        return 0

    else:
        alpha_dict2 = alpha_dict.copy()
        for i in range(26):
            # tmp = alpha_dict[s[i]]
            # del alpha_dict[s[i]]

            if s[i] != s_min[i]:
                remaining = [k for k, v in alpha_dict.items() if v > alpha_dict2[s[i - 1]]]

                if s[i:] == s_max[:26 - i]:
                    out = s[:i - 1] + remaining[0]
                    print(out)
                    return 0
                else:
                    out = s[:i + 1] + remaining[1]
                    print(out)
                    return 0

        print("abcdefghijklmnopqrstuvwxz")


def arc094_c():
    abc = list(map(int, input().split()))
    abc.sort()

    count = abc[2] - abc[1]
    diff = abc[2] - (abc[0] + count)
    count += [diff // 2 + 2, diff // 2][diff % 2 == 0]

    return count


def abc049_c():
    s = input()
    s = s.replace("eraser", "")
    s = s.replace("erase", "")
    s = s.replace("dreamer", "")
    s = s.replace("dream", "")
    # この順番に削除すること
    return ["NO", "YES"][s == ""]


def abc081_a():
    return sum(map(int, input()))


def abc081_b():
    input()
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
    return min(counts)


def abc083_b():
    n, a, b = map(int, input().split())
    num_sum = 0
    for i in range(1, n + 1):
        tmp = i
        dig_sum = 0
        while i:
            dig_sum += i % 10
            i //= 10
        if a <= dig_sum <= b:
            num_sum += tmp
    return num_sum


def abc085_b():
    n = int(input())
    d = []
    for i in range(n):
        d.append(int(input()))
    d = list(set(d))
    return len(d)


def abc085_c():
    n, y = map(int, input().split())

    for i in range(y // 10000 + 1):
        for j in range((y - 10000 * i) // 5000 + 1):
            if 10000 * i + 5000 * j + 1000 * (n - i - j) == y:
                return i, j, n - i - j
    return -1, -1, -1


def abc086_a():
    a, b = map(int, input().split())
    return ["Even", "Odd"][a * b % 2]


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
            return "No"
    return "Yes"


def abc087_b():
    a, b, c, x = map(int, input())
    count = 0
    for i in range(a + 1):
        for j in range(b + 1):
            for k in range(c + 1):
                if 500 * i + 100 * j + 50 * k == x:
                    count += 1
    return count


def abc088_b():
    input()
    a = list(map(int, input().split()))
    a.sort()
    a = a[::-1]
    alice = a[::2]
    bob = a[1::2]
    return sum(alice) - sum(bob)


def abc089_a():
    return int(input()) // 3


def abc089_b():
    input()
    colors = input()
    return ["Four", "Three"][colors.find("Y") == -1]


def abc089_c():
    num = int(input())
    namelist = []
    count_m = 0
    count_a = 0
    count_r = 0
    count_c = 0
    count_h = 0

    for _ in range(num):
        name = input()
        namelist.append(name)
        if name[0] == "M":
            count_m += 1
        elif name[0] == "A":
            count_a += 1
        elif name[0] == "R":
            count_r += 1
        elif name[0] == "C":
            count_c += 1
        elif name[0] == "H":
            count_h += 1
    print(count_m * count_a * count_r +
          count_m * count_a * count_c +
          count_m * count_a * count_h +
          count_m * count_r * count_c +
          count_m * count_r * count_h +
          count_m * count_c * count_h +
          count_a * count_r * count_c +
          count_a * count_r * count_h +
          count_a * count_c * count_h +
          count_r * count_c * count_h)


def abc089_d():
    hwd = list(map(int, input().split()))
    area = []
    lrs = []
    x_now = 0
    y_now = 0

    for i in range(0, hwd[0]):
        _aij = str(input()).split()
        area.append(list(map(int, _aij)))

    _q = int(input())

    for i in range(0, _q):
        _lr = str(input()).split()
        lrs.append(list(map(int, _lr)))

    for i in range(0, _q):
        _mpcount = 0
        for i2 in range(0, int((lrs[i][1] - lrs[i][0]) / hwd[2]) + 1):
            for i3 in range(0, hwd[0]):
                if lrs[i][0] + hwd[2] * i2 in area[i3]:
                    if not i2:
                        x_now = area[i3].index(lrs[i][0]) + 1
                        y_now = i3 + 1
                    _nextx = area[i3].index(lrs[i][0] + hwd[2] * i2) + 1
                    _nexty = i3 + 1
            _mpcount += abs(_nextx - x_now) + abs(_nexty - y_now)
            x_now = _nextx
            y_now = _nexty
        print(_mpcount)


def abc091_a():
    a, b, c = map(int, input().split())
    return ["No", "Yes"][a + b >= c]


def abc091_b():
    from collections import Counter
    s_str = []
    t_str = []
    _max = 0
    n = int(input())
    for i in range(n):
        s_str.append(input())
    m = int(input())
    for i in range(m):
        t_str.append(input())
    s_str2 = list(set(s_str))
    for element in s_str2:
        score = Counter(s_str)[element] - Counter(t_str)[element]
        if score > _max:
            _max = score
    return _max


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
    return _count


def abc093a():
    s = input()
    return ["No", "Yes"][(("a" in s) + ("b" in s) + ("c" in s)) == 3]


def abc093b():
    a, b, k = map(int, input().split())
    if b - a + 1 <= 2 * k:
        return [i for i in range(a, b + 1)]
    else:
        return [i for i in range(a, a + k)] + [i for i in range(b - k + 1, b + 1)]


def abc093c():
    abc = sorted(list(map(int, input().split())))
    return abc[2] - abc[1] + (abc[1] - abc[0]) // 2 + 2 * ((abc[1] - abc[0]) % 2)


def abc093d():
    q = int(input())
    ab_list = []
    for i in range(q):
        ab_list.append(list(map(int, input().split())))
    for ab in ab_list:
        pass


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
    return min(count, n[1] - count)


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

    def ncr(_n, _r):
        """
        Calculate the number of combination (nCr = nPr/r!).
        The parameters need to meet the condition of n >= r >= 0.
        It returns 1 if r == 0, which means there is one pattern
        to choice 0 items out of the number of n.
        """
        _r = min(_r, _n - _r)
        numerator = 1
        denominator = 1
        for i in range(_r):
            numerator *= (_n - i)
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
    return 700 + 100 * input().count("o")


def abc095_b():
    n, x = map(int, input().split())
    m = []
    for i in range(n):
        m.append(int(input()))

    x -= sum(m)
    count = x // min(m)
    return n + count


def abc095_c():
    a, b, c, x, y = map(int, input().split())

    diff = abs(x - y)
    price1 = [(a + b) * min(x, y), 2 * c * min(x, y)][a + b >= c * 2] + diff * [a, b][x < y]
    return min(price1, max(x, y) * c * 2)


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

        print(xr)


def abc096_a():
    a, b = map(int, input().split())
    return a - (a > b)


def abc096_b():
    import numpy as np
    a, b, c = map(int, input().split())
    k = int(input())

    return a + b + c + np.max([a, b, c]) * 2 ** (k - 1)


def abc096_c():
    import numpy as np
    h, w = map(int, input().split())
    c = np.zeros([h + 2, w + 2], dtype=int)
    for i in range(h):
        row = "." + input() + "."
        for j in range(w + 2):
            c[i + 1][j] = [0, 1][row[j] == "#"]

    for i in range(1, h + 2):
        for j in range(1, w + 2):
            if c[i][j] == 1 and (np.sum([c[i - 1][j], c[i + 1][j], c[i][j - 1], c[i][j + 1]]) == 0):
                return "No"
    return "Yes"


def abc096_dx():
    n = int(input())
    primes = [2, 3]

    for num in range(5, 55555, 2):
        is_prime = True
        for i in range(1, len(primes)):
            if primes[i] ** 2 > num:
                break
            if num % primes[i] == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(num)

    ans = []
    for num in primes:
        if num % 5 == 1:
            ans.append(str(num))
        if len(ans) == n:
            break
    return " ".join(ans)
