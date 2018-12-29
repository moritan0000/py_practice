def practice_a():
    a = int(input())
    b, c = map(int, input().split())
    s = input()
    print(a + b + c, s)


# practice_a()


def abc000_a():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc000_a()


def abc000_b():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc000_b()


def abc000_c():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc000_c()


def abc000_d():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc000_d()


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
        print(0)

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
                    print(0)
                else:
                    out = s[:i + 1] + remaining[1]
                    print(out)
                    print(0)

        print("abcdefghijklmnopqrstuvwxz")


def agc024_a():
    a, b, c, k = map(int, input().split())
    if k == 0:
        print([a - b, "Unfair"][abs(a - b) > 1e+18])

    coef_sum = 2 ** (k - 1)
    coef = [[coef_sum // 3, coef_sum // 3 + 1], [coef_sum // 3 + 1, coef_sum // 3]][coef_sum % 3 == 1]
    a_k = coef[0] * a + coef[1] * (b + c)
    b_k = coef[0] * b + coef[1] * (a + c)
    ans = [a_k - b_k, "Unfair"][abs(a_k - b_k) > 1e+18]

    print(ans)


def agc030_a():
    a, b, c = map(int, input().split())
    if a + b >= (c - 1):
        print(b + c)
    else:
        print(b + (a + b) + 1)


# agc030_a()


def agc030_b():
    def right(a, b):
        if b >= a:
            return b - a
        else:
            return 10 - (a - b)

    def left(a, b):
        if b <= a:
            return a - b
        else:
            return 10 - (b - a)

    l, n = map(int, input().split())
    x = [int(input()) for _ in range(n)]
    d = 0
    now = 0
    while x:
        if right(now, x[0]) > left(now, x[-1]):
            d += right(now, x[0])
            now = x[0]
            del x[0]
        else:
            d += left(now, x[-1])
            now = x[-1]
            del x[-1]
    print(d)


# agc030_b()


def agc030_c():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# agc030_c()


def agc030_d():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# agc030_d()


def arc094_c():
    abc = list(map(int, input().split()))
    abc.sort()

    count = abc[2] - abc[1]
    diff = abc[2] - (abc[0] + count)
    count += [diff // 2 + 2, diff // 2][diff % 2 == 0]

    print(count)


def abc001_a():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc001_a()


def abc001_b():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc001_b()


def abc001_c():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc001_c()


def abc001_d():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc001_d()


def abc002_a():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc002_a()


def abc002_b():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc002_b()


def abc002_c():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc002_c()


def abc002_d():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc002_d()


def abc003_a():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc003_a()


def abc003_b():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc003_b()


def abc003_c():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc003_c()


def abc003_d():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc003_d()


def abc004_a():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc004_a()


def abc004_b():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc004_b()


def abc004_c():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc004_c()


def abc004_d():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc004_d()


def abc005_a():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc005_a()


def abc005_b():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc005_b()


def abc005_c():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc005_c()


def abc005_d():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc005_d()


def abc006_a():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc006_a()


def abc006_b():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc006_b()


def abc006_c():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc006_c()


def abc006_d():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc006_d()


def abc007_a():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc007_a()


def abc007_b():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc007_b()


def abc007_c():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc007_c()


def abc007_d():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc007_d()


def abc008_a():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc008_a()


def abc008_b():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc008_b()


def abc008_c():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc008_c()


def abc008_d():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc008_d()


def abc009_a():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc009_a()


def abc009_b():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc009_b()


def abc009_c():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc009_c()


def abc009_d():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc009_d()


def abc010_a():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc010_a()


def abc010_b():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc010_b()


def abc010_c():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc010_c()


def abc010_d():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc010_d()


def abc011_a():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc011_a()


def abc011_b():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc011_b()


def abc011_c():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc011_c()


def abc011_d():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc011_d()


def abc012_a():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc012_a()


def abc012_b():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc012_b()


def abc012_c():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc012_c()


def abc012_d():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc012_d()


def abc013_a():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc013_a()


def abc013_b():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc013_b()


def abc013_c():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc013_c()


def abc013_d():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc013_d()


def abc014_a():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc014_a()


def abc014_b():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc014_b()


def abc014_c():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc014_c()


def abc014_d():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc014_d()


def abc015_a():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc015_a()


def abc015_b():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc015_b()


def abc015_c():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc015_c()


def abc015_d():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc015_d()


def abc016_a():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc016_a()


def abc016_b():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc016_b()


def abc016_c():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc016_c()


def abc016_d():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc016_d()


def abc017_a():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc017_a()


def abc017_b():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc017_b()


def abc017_c():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc017_c()


def abc017_d():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc017_d()


def abc018_a():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc018_a()


def abc018_b():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc018_b()


def abc018_c():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc018_c()


def abc018_d():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc018_d()


def abc019_a():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc019_a()


def abc019_b():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc019_b()


def abc019_c():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc019_c()


def abc019_d():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc019_d()


def abc020_a():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc020_a()


def abc020_b():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc020_b()


def abc020_c():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc020_c()


def abc020_d():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc020_d()


def abc021_a():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc021_a()


def abc021_b():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc021_b()


def abc021_c():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc021_c()


def abc021_d():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc021_d()


def abc022_a():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc022_a()


def abc022_b():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc022_b()


def abc022_c():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc022_c()


def abc022_d():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc022_d()


def abc023_a():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc023_a()


def abc023_b():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc023_b()


def abc023_c():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc023_c()


def abc023_d():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc023_d()


def abc024_a():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc024_a()


def abc024_b():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc024_b()


def abc024_c():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc024_c()


def abc024_d():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc024_d()


def abc025_a():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc025_a()


def abc025_b():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc025_b()


def abc025_c():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc025_c()


def abc025_d():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc025_d()


def abc026_a():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc026_a()


def abc026_b():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc026_b()


def abc026_c():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc026_c()


def abc026_d():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc026_d()


def abc027_a():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc027_a()


def abc027_b():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc027_b()


def abc027_c():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc027_c()


def abc027_d():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc027_d()


def abc028_a():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc028_a()


def abc028_b():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc028_b()


def abc028_c():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc028_c()


def abc028_d():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc028_d()


def abc029_a():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc029_a()


def abc029_b():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc029_b()


def abc029_c():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc029_c()


def abc029_d():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc029_d()


def abc030_a():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc030_a()


def abc030_b():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc030_b()


def abc030_c():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc030_c()


def abc030_d():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc030_d()


def abc031_a():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc031_a()


def abc031_b():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc031_b()


def abc031_c():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc031_c()


def abc031_d():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc031_d()


def abc032_a():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc032_a()


def abc032_b():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc032_b()


def abc032_c():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc032_c()


def abc032_d():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc032_d()


def abc033_a():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc033_a()


def abc033_b():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc033_b()


def abc033_c():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc033_c()


def abc033_d():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc033_d()


def abc034_a():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc034_a()


def abc034_b():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc034_b()


def abc034_c():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc034_c()


def abc034_d():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc034_d()


def abc035_a():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc035_a()


def abc035_b():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc035_b()


def abc035_c():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc035_c()


def abc035_d():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc035_d()


def abc036_a():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc036_a()


def abc036_b():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc036_b()


def abc036_c():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc036_c()


def abc036_d():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc036_d()


def abc037_a():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc037_a()


def abc037_b():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc037_b()


def abc037_c():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc037_c()


def abc037_d():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc037_d()


def abc038_a():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc038_a()


def abc038_b():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc038_b()


def abc038_c():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc038_c()


def abc038_d():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc038_d()


def abc039_a():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc039_a()


def abc039_b():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc039_b()


def abc039_c():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc039_c()


def abc039_d():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc039_d()


def abc040_a():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc040_a()


def abc040_b():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc040_b()


def abc040_c():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc040_c()


def abc040_d():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc040_d()


def abc041_a():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc041_a()


def abc041_b():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc041_b()


def abc041_c():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc041_c()


def abc041_d():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc041_d()


def abc042_a():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc042_a()


def abc042_b():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc042_b()


def abc042_c():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc042_c()


def abc042_d():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc042_d()


def abc043_a():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc043_a()


def abc043_b():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc043_b()


def abc043_c():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc043_c()


def abc043_d():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc043_d()


def abc044_a():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc044_a()


def abc044_b():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc044_b()


def abc044_c():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc044_c()


def abc044_d():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc044_d()


def abc045_a():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc045_a()


def abc045_b():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc045_b()


def abc045_c():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc045_c()


def abc045_d():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc045_d()


def abc046_a():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc046_a()


def abc046_b():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc046_b()


def abc046_c():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc046_c()


def abc046_d():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc046_d()


def abc047_a():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc047_a()


def abc047_b():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc047_b()


def abc047_c():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc047_c()


def abc047_d():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc047_d()


def abc048_a():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc048_a()


def abc048_b():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc048_b()


def abc048_c():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc048_c()


def abc048_d():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc048_d()


def abc049_a():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc049_a()


def abc049_b():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc049_b()


def abc049_c():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc049_c()


def abc049_d():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc049_d()


def abc050_a():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc050_a()


def abc050_b():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc050_b()


def abc050_c():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc050_c()


def abc050_d():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc050_d()


def abc051_a():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc051_a()


def abc051_b():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc051_b()


def abc051_c():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc051_c()


def abc051_d():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc051_d()


def abc052_a():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc052_a()


def abc052_b():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc052_b()


def abc052_c():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc052_c()


def abc052_d():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc052_d()


def abc053_a():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc053_a()


def abc053_b():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc053_b()


def abc053_c():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc053_c()


def abc053_d():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc053_d()


def abc054_a():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc054_a()


def abc054_b():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc054_b()


def abc054_c():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc054_c()


def abc054_d():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc054_d()


def abc055_a():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc055_a()


def abc055_b():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc055_b()


def abc055_c():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc055_c()


def abc055_d():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc055_d()


def abc056_a():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc056_a()


def abc056_b():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc056_b()


def abc056_c():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc056_c()


def abc056_d():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc056_d()


def abc057_a():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc057_a()


def abc057_b():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc057_b()


def abc057_c():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc057_c()


def abc057_d():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc057_d()


def abc058_a():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc058_a()


def abc058_b():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc058_b()


def abc058_c():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc058_c()


def abc058_d():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc058_d()


def abc059_a():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc059_a()


def abc059_b():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc059_b()


def abc059_c():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc059_c()


def abc059_d():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc059_d()


def abc060_a():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc060_a()


def abc060_b():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc060_b()


def abc060_c():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc060_c()


def abc060_d():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc060_d()


def abc061_a():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc061_a()


def abc061_b():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc061_b()


def abc061_c():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc061_c()


def abc061_d():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc061_d()


def abc062_a():
    x, y = map(int, input().split())
    a = [1, 3, 5, 7, 8, 10, 12]
    b = [4, 6, 9, 11]
    c = [2]
    if (x in a and y in a) or (x in b and y in b) or (x in c and y in c):
        print("Yes")
    else:
        print("No")


# abc062_a()


def abc062_b():
    h, w = map(int, input().split())
    a = [None] * (h + 2)
    a[0] = a[-1] = ["#"] * (w + 2)
    for i in range(1, h + 1):
        a[i] = ["#"] + [input()] + ["#"]

    for ln in a:
        print("".join(ln))


# abc062_b()


def abc062_c():
    h, w = map(int, input().split())
    if h % 3 == 0 or w % 3 == 0:
        print(0)
    else:
        print(h, w)


# abc062_c()


def abc062_d():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc062_d()


def abc063_a():
    a, b = map(int, input().split())
    if a + b < 10:
        print(a + b)
    else:
        print("error")


# abc063_a()


def abc063_b():
    s = list(input())
    if len(s) == len(set(s)):
        print("yes")
    else:
        print("no")


# abc063_b()


def abc063_c():
    from itertools import combinations
    import numpy as np
    n = int(input())
    s = np.array([int(input()) for _ in range(n)])
    score = 0
    if sum(s % 10) == 0:
        print(0)
    else:
        for i in range(n + 1, 0, -1):
            for seq in combinations(s, i):
                tmp = sum(seq)
                if tmp % 10 and tmp > score:
                    score = tmp
            if score:
                break
        print(score)


# abc063_c()


def abc063_d():
    import numpy as np
    n, a, b = map(int, input().split())
    h = np.array([int(input()) for _ in range(n)])
    diff = a - b
    count = 0

    while any(h > 0):
        hmax_index = np.argmax(h)
        h -= b
        h[hmax_index] -= diff
        h = h[h > 0]
        count += 1

    print(count)


# abc063_d()


def abc064_a():
    r, g, b = input().split()
    if int(r + g + b) % 4 == 0:
        print("YES")
    else:
        print("NO")


# abc064_a()


def abc064_b():
    _ = int(input())
    a = list(map(int, input().split()))
    print(max(a) - min(a))


# abc064_b()


def abc064_c():
    _ = int(input())
    a = list(map(int, input().split()))
    color = [False] * 8
    higher = 0
    for v in a:
        if v >= 3200:
            higher += 1
        else:
            color[v // 400] = True

    print(max(1, sum(color)), sum(color) + higher)


# abc064_c()


def abc064_d():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc064_d()


def abc065_a():
    x, a, b = map(int, input().split())
    if b <= a:
        print("delicious")
    elif b <= a + x:
        print("safe")
    else:
        print("dangerous")


# abc065_a()


def abc065_b():
    n = int(input())
    a = [int(input()) for _ in range(n)]
    now = 1
    count = 0
    for i in range(n):
        if now == 2:
            print(count)
            return None
        now = a[now - 1]
        count += 1
    print(-1)


# abc065_b()


def abc065_c():
    from math import factorial
    n, m = map(int, input().split())
    if abs(n - m) >= 2:
        print(0)
    elif n == m:
        print((2 * factorial(n) * factorial(m)) % (10 ** 9 + 7))
    else:
        print((factorial(n) * factorial(m)) % (10 ** 9 + 7))


# abc065_c()


def abc065_d():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc065_d()


def abc066_a():
    a, b, c = map(int, input().split())
    print(a + b + c - max([a, b, c]))


# abc066_a()


def abc066_b():
    s = input()
    for i in range(len(s) // 2):
        s = s[:-2]
        if s[:len(s) // 2] == s[len(s) // 2:]:
            print(len(s))
            break


# abc066_b()


def abc066_c():
    n = int(input())
    a = input().split()
    b = a[1::2][::-1] + a[::2]
    print(" ".join(b[::(-1) ** (n % 2)]))


# abc066_c()


def abc066_d():
    n = int(input())
    a = list(map(int, input().split()))
    print(n, a)
    for i in range(1, n + 2):
        print()


# abc066_d()


def abc067_a():
    a, b = map(int, input().split())
    if a * b * (a + b) % 3 == 0:
        print("Possible")
    else:
        print("Impossible")


# abc067_a()


def abc067_b():
    _, k = map(int, input().split())
    ls = sorted(list(map(int, input().split())), reverse=True)
    print(sum(ls[:k]))


# abc067_b()


def abc067_c():
    _ = int(input())
    a = list(map(int, input().split()))
    x = a[0]
    y = sum(a[1:])
    del a[0]
    del a[-1]
    diff = abs(x - y)
    for v in a:
        x += v
        y -= v
        if abs(x - y) < diff:
            diff = abs(x - y)
    print(diff)


# abc067_c()


def abc067_d():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc067_d()


def abc068_a():
    n = input()
    print("ABC" + n)


# abc068_a()


def abc068_b():
    n = int(input())
    ans = 1
    count = 0

    for i in range(1, n + 1):
        num = i
        tmp_count = 0
        while num % 2 == 0:
            tmp_count += 1
            num /= 2
        if tmp_count > count:
            ans = i
            count = tmp_count

    print(ans)


# abc068_b()


def abc068_c():
    from collections import Counter
    n, m = map(int, input().split())
    ab = [list(map(int, input().split())) for _ in range(m)]
    candidates = []

    for i in range(m):
        if ab[i][0] == 1:
            candidates.append(ab[i][1])
        elif ab[i][1] == n:
            candidates.append(ab[i][0])

    if max(Counter(candidates).values()) >= 2:
        print("POSSIBLE")
    else:
        print("IMPOSSIBLE")


# abc068_c()


def abc068_d():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc068_d()


def abc069_a():
    n, m = map(int, input().split())
    print((n - 1) * (m - 1))


# abc069_a()


def abc069_b():
    s = input()
    print(s[0] + str(len(s) - 2) + s[-1])


# abc069_b()


def abc069_c():
    n = int(input())
    a = list(map(int, input().split()))
    two = 0
    four = 0
    odd = 0
    for v in a:
        if v % 2 == 0:
            if v % 4 == 0:
                four += 1
            else:
                two += 1
        else:
            odd += 1

    if four >= odd or (four + odd == n and odd - four == 1):
        print("Yes")
    else:
        print("No")


# abc069_c()


def abc069_d():
    h, w = map(int, input().split())
    n = int(input())
    a = list(map(int, input().split()))
    s = []
    for i in range(n):
        s.extend([str(i + 1)] * a[i])

    for i in range(h):
        if i % 2 == 0:
            print(" ".join(s[i * w:(i + 1) * w]))
        else:
            print(" ".join(s[i * w:(i + 1) * w][::-1]))


# abc069_d()


def abc070_a():
    n = input()
    if n[0] == n[2]:
        print("Yes")
    else:
        print("No")


# abc070_a()


def abc070_b():
    a, b, c, d = map(int, input().split())
    print(max([0, min(b, d) - max(a, c)]))


# abc070_b()


def abc070_c():
    from fractions import gcd
    n = int(input())
    t = [int(input()) for _ in range(n)]
    ans = t[0]
    for v in t:
        ans = ans * v // gcd(ans, v)

    print(ans)


# abc070_c()


def abc070_d():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc070_d()


def abc071_a():
    x, a, b = map(int, input().split())
    if abs(x - a) < abs(x - b):
        print("A")
    else:
        print("B")


# abc071_a()


def abc071_b():
    s = input()
    alpha = [chr(i) for i in range(ord("a"), ord("z") + 1)]
    for char in set(s):
        alpha.remove(char)

    if alpha:
        print(alpha[0])
    else:
        print("None")


# abc071_b()


def abc071_c():
    from collections import Counter
    _ = int(input())
    a = list(map(int, input().split()))
    a_count = Counter(a)
    ans = 1

    if len(a_count) == 1:
        print(a[0] ** 2)
    elif a_count.most_common()[1][1] < 2:
        print(0)
    else:
        tmp = 0
        for v in sorted(set(a), reverse=True):
            if tmp == 2:
                break
            if tmp == 0 and a_count[v] >= 4:
                ans = v ** 2
                break
            if a_count[v] >= 2:
                ans *= v
                tmp += 1
        print(ans)


# abc071_c()


def abc071_d():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc071_d()


def abc072_a():
    x, t = map(int, input().split())
    print(max(x - t, 0))


# abc072_a()


def abc072_b():
    s = input()
    print("".join(s[::2]))


# abc072_b()


def abc072_c():
    from collections import Counter

    _ = int(input())
    a = list(map(int, input().split()))
    a_ex = []
    for num in a:
        a_ex.extend([num - 1, num, num + 1])

    print(max(Counter(a_ex).values()))


# abc072_c()


def abc072_d():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc072_d()


def abc073_a():
    print(["No", "Yes"]["9" in input()])


# abc073_a()


def abc073_b():
    n = int(input())
    tot = 0
    for _ in range(n):
        l, r = map(int, input().split())
        tot += (r - l) + 1
    print(tot)


# abc073_b()


def abc073_c():
    from collections import Counter
    n = int(input())
    a = [int(input()) for _ in range(n)]
    a_count = Counter(a)
    ans = 0

    for num in a_count:
        if a_count[num] % 2:
            ans += 1
    print(ans)


# abc073_c()


def abc073_d():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc073_d()


def abc074_a():
    n = int(input())
    a = int(input())
    print(n ** 2 - a)


# abc074_a()


def abc074_b():
    n = int(input())
    k = int(input())
    x = list(map(int, input().split()))
    d = 0
    for i in range(n):
        d += 2 * min(x[i], abs(k - x[i]))
    print(d)


# abc074_b()


def abc074_c():
    from math import ceil
    a, b, c, d, e, f = map(int, input().split())
    ans_wat = 100 * a
    ans_sug = 0

    for i in range(ceil(f / (100 * a))):
        for j in range(ceil((f - 100 * a * i) / (100 * b))):
            for k in range((f - 100 * (a * i + b * j)) // c + 1):
                for l in range((f - 100 * (a * i + b * j) - c * k) // d + 1):
                    tmp_sug = c * k + d * l
                    tmp_wat = 100 * (a * i + b * j)
                    if tmp_wat and tmp_sug / tmp_wat <= e / 100 and \
                            tmp_sug / (tmp_wat + tmp_sug) > ans_sug / (ans_wat + ans_sug):
                        ans_sug = c * k + d * l
                        ans_wat = 100 * (a * i + b * j)
    print(ans_wat + ans_sug, ans_sug)


# abc074_c()


def abc074_d():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc074_d()


def abc075_a():
    a, b, c = map(int, input().split())
    if a == b:
        print(c)
    elif b == c:
        print(a)
    else:
        print(b)


# abc075_a()


def abc075_b():
    h, w = map(int, input().split())
    s = [None] * (h + 2)
    ans = [[None] * w for _ in range(h)]

    s[0] = s[-1] = ["."] * (w + 2)
    for i in range(1, h + 1):
        s[i] = ["."] + list(input()) + ["."]

    for i in range(1, h + 1):
        for j in range(1, w + 1):
            if s[i][j] == "#":
                ans[i - 1][j - 1] = "#"
            else:
                ans[i - 1][j - 1] = str([s[i - 1][j - 1], s[i - 1][j], s[i - 1][j + 1], s[i][j - 1], s[i][j + 1],
                                         s[i + 1][j - 1], s[i + 1][j], s[i + 1][j + 1]].count("#"))

    for ln in ans:
        print("".join(ln))


# abc075_b()


def abc075_c():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc075_c()


def abc075_d():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc075_d()


def abc076_a():
    r = int(input())
    g = int(input())
    print(r + 2 * (g - r))


# abc076_a()


def abc076_b():
    n = int(input())
    k = int(input())
    ans = 1
    for _ in range(n):
        ans = min(ans * 2, ans + k)
    print(ans)


# abc076_b()


def abc076_c():
    s = input()
    t = input()

    if len(t) > len(s):
        print("UNRESTORABLE")
    else:
        for i in range(len(t), len(s) + 1):
            flag = False
            for j in range(len(t)):
                if t[j] not in ["?", s[-i + j]]:
                    flag = True
                    break
            if not flag:
                ans = s[:-i + 1] + t + s[-i + len(t)]
                print(ans)
                return 0
        print("UNRESTORABLE")


# abc076_c()


def abc076_d():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc076_d()


def abc077_a():
    c = [list(input()) for _ in range(2)]
    if c[0][0] == c[1][2] and c[0][1] == c[1][1] and c[0][2] == c[1][0]:
        print("YES")
    else:
        print("NO")


# abc077_a()


def abc077_b():
    n = int(input())
    print(int(n ** 0.5) ** 2)


# abc077_b()


def abc077_c():
    n = int(input())
    a, b, c = [sorted(list(map(int, input().split()))) for _ in range(3)]
    print(n, a, b, c)


# abc077_c()


def abc077_d():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc077_d()


def abc078_a():
    x, y = input().split()
    if ord(x) < ord(y):
        print("<")
    elif ord(x) > ord(y):
        print(">")
    else:
        print("=")


# abc078_a()


def abc078_b():
    x, y, z = map(int, input().split())
    print((x - z) // (y + z))


# abc078_b()


def abc078_c():
    n, m = map(int, input().split())
    time1 = 1900 * m + 100 * (n - m)
    print(time1 * 2 ** m)


# abc078_c()


def abc078_d():
    n, z, w = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, z, w, a)


# abc078_d()


def abc079_a():
    n = input()
    if int(n[1:]) % 111 == 0 or int(n[:-1]) % 111 == 0:
        print("Yes")
    else:
        print("No")


# abc079_a()


def abc079_b():
    n = int(input())
    if n == 1:
        print(1)
    else:
        lucas = [0] * (n + 1)
        lucas[0] = 2
        lucas[1] = 1
        for i in range(2, n + 1):
            lucas[i] = lucas[i - 1] + lucas[i - 2]
        print(lucas[n])


# abc079_b()


def abc079_c():
    n = list(map(int, input()))
    a = n[0]
    for b in [n[1], -n[1]]:
        for c in [n[2], -n[2]]:
            for d in [n[3], -n[3]]:
                if a + b + c + d == 7:
                    op1 = ["+", "-"][b < 0]
                    op2 = ["+", "-"][c < 0]
                    op3 = ["+", "-"][d < 0]
                    print(a, op1, abs(b), op2, abs(c), op3, abs(d), "=7", sep="")
                    return 0


# abc079_c()


def abc079_d():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc079_d()


def abc080_a():
    n, a, b = map(int, input().split())
    print(min(n * a, b))


# abc080_a()


def abc080_b():
    n = x = int(input())
    dig_sum = 0

    while n:
        dig_sum += n % 10
        n //= 10

    if x % dig_sum:
        print("No")
    else:
        print("Yes")


# abc080_b()


def abc080_c():
    import numpy as np
    import itertools

    n = int(input())
    f = np.array([list(map(int, input().split())) for _ in range(n)])
    p = np.array([list(map(int, input().split())) for _ in range(n)])
    benefits = np.array([], dtype=int)

    for i in range(1, 11):
        for seq in itertools.combinations([i for i in range(10)], i):
            tmp = 0
            for j in range(n):
                tmp += p[j][sum(f[j][np.array(seq)])]
            benefits = np.append(benefits, tmp)

    print(max(benefits))


# abc080_c()


def abc080_d():
    n = int(input())
    x, y = map(int, input().split())
    a = list(map(int, input().split()))
    print(n, x, y, a)


# abc080_d()


def abc081_a():
    s = input()
    print(s.count("1"))


# abc081_a()


def abc081_b():
    _ = int(input())
    a = list(map(int, input().split()))
    ans = 100000
    for num in a:
        tmp = 0
        while num % 2 == 0:
            num /= 2
            tmp += 1
        if tmp < ans:
            ans = tmp
    print(ans)


# abc081_b()


def abc081_c():
    from collections import Counter
    n, k = map(int, input().split())
    a = list(map(int, input().split()))

    a_count = Counter(a)
    if len(a_count) <= k:
        print(0)
    else:
        val = list(a_count.values())
        val.sort()
        print(sum(val[:-k]))


# abc081_c()


def abc081_d():
    n = int(input())
    a = list(map(int, input().split()))
    print(n, a)


# abc081_d()


def abc082_a():
    from math import ceil
    a, b = map(int, input().split())
    print(ceil((a + b) / 2))


# abc082_a()


def abc082_b():
    s = sorted(input())
    t = sorted(input(), reverse=True)

    for i in range(min(len(s), len(t))):
        if ord(s[i]) < ord(t[i]):
            print("Yes")
            return 0

    if len(s) < len(t):
        for c in s:
            if c in t:
                t.remove(c)
            else:
                print("No")
                return 0
        print("Yes")
        return 0

    print("No")


# abc082_b()


def abc082_c():
    from collections import Counter

    _ = int(input())
    a = list(map(int, input().split()))
    ans = 0
    a_count = Counter(a)

    for num in a_count:
        if num <= a_count[num]:
            ans += (a_count[num] - num)
        else:
            ans += a_count[num]

    print(ans)


# abc082_c()


def abc082_d():
    s = input()
    x, y = map(int, input().split())

    print(s, x, y)


# abc082_d()


def abc083_a():
    a, b, c, d = map(int, input().split())
    if a + b > c + d:
        print("Left")
    elif a + b < c + d:
        print("Right")
    else:
        print("Balanced")


# abc083_a()


def abc083_b():
    n, a, b = map(int, input().split())
    ans = 0
    for i in range(1, n + 1):
        tmp = 0
        for digit in str(i):
            tmp += int(digit)
        if a <= tmp <= b:
            ans += i
    print(ans)


# abc083_b()


def abc083_c():
    x, y = map(int, input().split())
    n = 1
    while x * 2 <= y:
        n += 1
        x *= 2
    print(n)


# abc083_c()


def abc083_d():
    n = int(input())
    print(n)


# abc083_d()


def abc084_a():
    m = int(input())
    print(48 - m)


# abc084_a()


def abc084_b():
    a, b = map(int, input().split())
    s = input()

    if "-" in s[:a] or s[a] != "-" or "-" in s[-b:]:
        print("No")
    else:
        print("Yes")


# abc084_b()


def abc084_c():
    from math import ceil
    n = int(input())
    c, s, f = [[None] * (n - 1) for _ in range(3)]
    for i in range(n - 1):
        c[i], s[i], f[i] = map(int, input().split())

    for i in range(n - 1):
        time = 0
        for j in range(i, n - 1):
            if time < s[j]:
                time = s[j]
            else:
                time = f[j] * ceil(time / f[j])
            time += c[j]
        print(time)
    print(0)


# abc084_c()


def abc084_d():
    import numpy as np

    def prime_list(limit):
        if limit == 1:
            return []

        prime_nums = [2]
        for _i in range(3, limit + 1, 2):
            factor = False
            for divisor in range(3, int(_i ** 0.5) + 1, 2):
                if _i % divisor == 0:
                    factor = True

            if not factor:
                prime_nums.append(_i)

        return prime_nums

    q = int(input())
    l, r = [[None] * q for _ in range(2)]
    for i in range(q):
        l[i], r[i] = map(int, input().split())

    primes = np.array(prime_list(max(r)))
    for i in range(q):
        ans = 0
        candidates = primes[(l[i] <= primes) & (primes <= r[i])]
        for val in candidates:
            if (val + 1) // 2 in primes:
                ans += 1
        print(ans)


# abc084_d()


def abc085_a():
    s = input()
    print(s[:2] + "18" + s[4:])


# abc085_a()


def abc085_b():
    n = int(input())
    d = [int(input()) for _ in range(n)]
    print(len(set(d)))


# abc085_b()


def abc085_c():
    n, y = map(int, input().split())
    for i in range(n + 1):
        if 10000 * i + 5000 * (n - i) < y:
            continue
        for j in range(n - i + 1):
            if 10000 * i + 5000 * j + 1000 * (n - i - j) == y:
                print(i, j, n - i - j)
                return 0
    print("-1 -1 -1")


# abc085_c()


def abc085_d():
    n = int(input())
    print(n)


# abc085_d()


def abc086_a():
    a, b = map(int, input().split())
    if a * b % 2 == 1:
        print("Odd")
    else:
        print("Even")


# abc086_a()


def abc086_b():
    a, b = input().split(" ")
    n = int(a + b)
    if int((n ** 0.5)) ** 2 == n:
        print("Yes")
    else:
        print("No")


# abc086_b()


def abc086_c():
    n = int(input())
    t = [0] * (n + 1)
    x = [0] * (n + 1)
    y = [0] * (n + 1)
    for i in range(1, n + 1):
        t[i], x[i], y[i] = map(int, input().split())

    for i in range(n):
        if abs(x[i + 1] - x[i]) + abs(y[i + 1] - y[i]) > (t[i + 1] - t[i]) or \
                abs(abs(x[i + 1] - x[i]) + abs(y[i + 1] - y[i]) - (t[i + 1] - t[i])) % 2 == 1:
            print("No")
            return 0
    print("Yes")


# abc086_c()


def abc086_d():
    n = int(input())
    print(n)


# abc086_d()


def abc087_a():
    x = int(input())
    a = int(input())
    b = int(input())
    print((x - a) % b)


# abc087_a()


def abc087_b():
    a = int(input())
    b = int(input())
    c = int(input())
    x = int(input())
    count = 0

    for i in range(a + 1):
        for j in range(b + 1):
            for k in range(c + 1):
                if (500 * i + 100 * j + 50 * k) == x:
                    count += 1
    print(count)


# abc087_b()


def abc087_c():
    n = int(input())
    a = [list(map(int, input().split())) for _ in range(2)]
    candy = 0

    for i in range(n):
        num = sum(a[0][:i + 1]) + sum(a[1][i:])
        if num > candy:
            candy = num
    print(candy)


# abc087_c()


def abc087_d():
    n, m = map(int, input().split())
    print(n, m)


# abc087_d()


def abc088_a():
    n = int(input())
    a = int(input())
    if n % 500 <= a:
        print("Yes")
    else:
        print("No")


def abc088_b():
    _ = int(input())
    a = list(map(int, input().split()))
    a.sort(reverse=True)
    print(sum(a[::2]) - sum(a[1::2]))


def abc088_c():
    c = [list(map(int, input().split())) for _ in range(3)]
    print(c)


# abc088_c()


def abc088_d():
    n = int(input())
    print(n)


# abc088_d()


def abc089_a():
    print(int(input()) // 3)


def abc089_b():
    input()
    colors = input()
    print(["Four", "Three"][colors.find("Y") == -1])


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
    _x_now = 0
    _y_now = 0

    for i in range(0, hwd[0]):
        _aij = str(input()).split()
        area.append(list(map(int, _aij)))

    _q = int(input())

    for i in range(0, _q):
        _lr = str(input()).split()
        lrs.append(list(map(int, _lr)))

    # for i in range(0, _q):
    #     _mpcount = 0
    #     for i2 in range(0, int((lrs[i][1] - lrs[i][0]) / hwd[2]) + 1):
    #         for i3 in range(0, hwd[0]):
    #             if lrs[i][0] + hwd[2] * i2 in area[i3]:
    #                 if not i2:
    #                     x_now = area[i3].index(lrs[i][0]) + 1
    #                     y_now = i3 + 1
    #                 _nextx = area[i3].index(lrs[i][0] + hwd[2] * i2) + 1
    #                 _nexty = i3 + 1
    #         _mpcount += abs(_nextx - x_now) + abs(_nexty - y_now)
    #         x_now = _nextx
    #         y_now = _nexty
    #     print(_mpcount)


def abc090_a():
    c = [input() for _ in range(3)]
    print(c[0][0] + c[1][1] + c[2][2])


def abc090_b():
    a, b = map(int, input().split())
    num = 0
    for i in range(a, b + 1):
        s = str(i)
        if s[0] == s[-1] and s[1] == s[-2]:
            num += 1
    print(num)


def abc090_c():
    n, m = map(int, input().split())
    print(abs((n - 2) * (m - 2)))


def abc090_d():
    n, k = map(int, input().split())
    num = 0
    for b in range(1, n):
        num += abs(b - k) * (n // b)

    print(num)


# abc090_d()


def abc091_a():
    a, b, c = map(int, input().split())
    print(["No", "Yes"][a + b >= c])


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


def abc092_a():
    a = int(input())
    b = int(input())
    c = int(input())
    d = int(input())
    print(min(a, b) + min(c, d))


def abc092_b():
    n = int(input())
    d, x = map(int, input().split())
    a = [int(input()) for _ in range(n)]
    choco = x
    for i in range(n):
        choco += (d - 1) // a[i] + 1

    print(choco)


def abc092_c():
    n = int(input())
    a = [0] + list(map(int, input().split())) + [0]
    money = 0
    for i in range(n + 1):
        money += abs(a[i] - a[i + 1])
    for i in range(1, n + 1):
        print(money - abs(a[i] - a[i - 1]) - abs(a[i] - a[i + 1]) + abs(a[i + 1] - a[i - 1]))


def abc092_d():
    a, b = map(int, input().split())
    print(a, b)


# abc092_d()


def abc093a():
    s = input()
    print(["No", "Yes"][(("a" in s) + ("b" in s) + ("c" in s)) == 3])


def abc093b():
    a, b, k = map(int, input().split())
    if b - a + 1 <= 2 * k:
        print([i for i in range(a, b + 1)])
    else:
        print([i for i in range(a, a + k)] + [i for i in range(b - k + 1, b + 1)])


def abc093c():
    abc = sorted(list(map(int, input().split())))
    print(abc[2] - abc[1] + (abc[1] - abc[0]) // 2 + 2 * ((abc[1] - abc[0]) % 2))


def abc093d():
    q = int(input())
    ab_list = []
    for i in range(q):
        ab_list.append(list(map(int, input().split())))
    print(0)


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
        print(0)

    a = np.array(a)
    ai = int(a[-1])

    if n % 2 == 0:
        r = ai / 2
        b = a.copy() - r
        j = np.argmin(abs(b))
        print(int(ai), int(a[j]))
        print(0)
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
        print(0)


def abc095_a():
    print(700 + 100 * input().count("o"))


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


def abc095_d():
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
    print(a - (a > b))


def abc096_b():
    import numpy as np
    a, b, c = map(int, input().split())
    k = int(input())

    print(a + b + c + np.max([a, b, c]) * 2 ** (k - 1))


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
                print("No")
    print("Yes")


def abc096_d():
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
    print(" ".join(ans))


def abc097_a():
    a, b, c, d = map(int, input().split())
    print(["No", "Yes"][(abs(a - c) <= d) or ((a - b) <= d and (c - b) <= d)])


def abc097_b():
    x = int(input())
    if x == 1:
        print(1)

    n = int(x ** 0.5)
    ans = []
    for i in range(2, n + 1):
        tmp = i
        while tmp * i <= x:
            tmp *= i
        ans.append(tmp)

    print(max(ans))


def abc097_c():
    import numpy as np
    s = input()
    k = int(input())

    substr = [[None] * len(s)]
    substr[-1] = np.array([s[-1]])

    for i in range(len(s) - 2, -1, -1):
        # substr[i] =
        # (np.append(np.array([s[i]], dtype=object) + np.array(substr[i + 1], dtype=object), s[i])).astype(str)

        substr[i] = [s[i] + val for val in substr[i + 1]] + [s[i]]
    # substr = np.array(substr).flatten()
    substr = np.unique([val for vals in substr for val in vals])
    substr.sort()

    print(substr[k - 1])


def abc097_d():
    n, m = map(int, input().split())
    p = map(int, input().split())
    xy = [list(map(int, input().split())) for _ in range(m)]

    print(p, xy)


def abc098_a():
    a, b = map(int, input().split())
    print(max(a + b, a - b, a * b))


def abc098_b():
    n = int(input())
    s = input()
    count_max = 0
    if len(s) == 2:
        print([0, 1][s[0] == s[1]])
    for i in range(1, n - 1):
        x = set(s[:i])
        y = set(s[i:])
        count = 0
        for let in x:
            if let in y:
                count += 1
        count_max = max(count, count_max)
    print(count_max)


def abc098_c():
    n = int(input())
    s = input()
    count_min = n - 1
    w_num = s.count("W")
    for i in range(n):
        w_left = s[:i].count("W")
        e_right = n - w_num - (i - w_left) - (s[i] == "E")
        count = w_left + e_right
        if count == 0:
            print(0)
        count_min = min(count, count_min)

    print(count_min)


def abc098_d():
    n = int(input())
    a = list(map(int, input().split()))
    count = 0
    for l in range(n):
        for r in range(l, n):
            xor = a[l]
            for k in range(l + 1, r + 1):
                xor ^= a[k]
            count += sum(a[l:r + 1]) == xor
    print(count)


def abc099_a():
    n = int(input())

    print(["ABC", "ABD"][n >= 1000])


def abc099_b():
    a, b = map(int, input().split())
    x = sum([i for i in range(b - a)])
    print(x - a)


def abc099_c():
    n = int(input())
    if n == 3:
        print(3)
    vals = [6, 9, 36, 81, 216, 729, 1296, 6561, 7776, 46656, 59049][::-1]

    count = n % 3
    n -= count

    while n:
        for val in vals:
            if n >= val and (n - val) != 3:
                n -= val
                count += 1
                break

    print(count)


def abc099_d():
    print(0)


def abc100_a():
    a, b = map(int, input().split())
    print([":(", "Yay!"][max(a, b) <= 8])


def abc100_b():
    d, n = map(int, input().split())
    if n <= 99:
        print(100 ** d * n)
    else:
        print(100 ** (d + 1) + 100 ** d)


def abc100_c():
    _ = int(input())
    a = list(map(int, input().split()))
    ans = 0
    for val in a:
        while val % 2 == 0:
            ans += 1
            val /= 2
    print(ans)


def abc100_d():
    import itertools

    n, m = map(int, input().split())
    x = [0] * n
    y = [0] * n
    z = [0] * n
    ans = -10000000000 * 3
    for i in range(n):
        x[i], y[i], z[i] = map(int, input().split())

    for seq in itertools.combinations([i for i in range(n)], m):
        tmp_x = 0
        tmp_y = 0
        tmp_z = 0
        for i in seq:
            tmp_x += x[i]
            tmp_y += y[i]
            tmp_z += z[i]
        tmp_sum = abs(tmp_x) + abs(tmp_y) + abs(tmp_z)
        ans = max(tmp_sum, ans)

    print(ans)


# abc100_d()


def abc101_a():
    s = input()
    ans = s.count("+") - s.count("-")
    print(ans)


def abc101_b():
    n = int(input())
    tmp = n
    sn = 0
    while tmp:
        sn += tmp % 10
        tmp //= 10
    ans = ["No", "Yes"][n % sn == 0]
    print(ans)


def abc101_c():
    n, k = map(int, input().split())
    _ = list(map(int, input().split()))
    ans = (n - 1) // (k - 1) + [1, 0][(n - 1) % (k - 1) == 0]
    print(ans)


def abc101_d():
    k = int(input())

    snuke_nums = []
    for i in range(15):
        snuke_nums.extend([n * (10 ** i) - 1 for n in range(2, 11)])
        if len(snuke_nums) >= k:
            break

    num = len(snuke_nums)
    snuke_sn = [0] * num
    for i in range(num):
        tmp = snuke_nums[i]
        sn = 0
        while tmp:
            sn += tmp % 10
            tmp //= 10
        snuke_sn[i] = snuke_nums[i] / sn

    snuke_sn_sort = sorted(snuke_sn)

    snuke_flag = [False] * num
    for i in range(num):
        if snuke_sn[i] != snuke_sn_sort[i]:
            snuke_flag[i] = True

    snuke_ans = []
    for i in range(num):
        if not snuke_flag[i]:
            snuke_ans.append(snuke_nums[i])

    for i in range(k):
        print(snuke_ans[i])


def abc102_a():
    n = int(input())
    print([n, 2 * n][n % 2])


def abc102_b():
    _ = int(input())
    a = list(map(int, input().split()))

    print(max(a) - min(a))


def abc102_c():
    n = int(input())
    print(n)


# abc102_c()


def abc102_d():
    n = int(input())
    print(n)


# abc102_d()


def abc103_a():
    a = list(map(int, input().split()))
    cost = [abs(a[0] - a[1]), abs(a[1] - a[2]), abs(a[2] - a[0])]
    cost.sort()
    print(sum(cost[:-1]))


def abc103_b():
    s = input()
    t = input()

    for _ in range(len(s)):
        if s == t:
            print("Yes")
            return 0
        else:
            s = s[-1] + s[:-1]
    print("No")


def abc103_c():
    n = int(input())
    a = list(map(int, input().split()))
    print(sum(a) - n)


def abc103_d():
    n = int(input())
    print(n)


# abc103_d()


def abc104_a():
    r = int(input())
    if r < 1200:
        print("ABC")
    elif r < 2800:
        print("ARC")
    else:
        print("AGC")


def abc104_b():
    s = input()
    num = s.find("C")
    if s[0] == "A" and s[2:-1].count("C") == 1 and (s[1:num] + s[num + 1:]) == (s[1:num] + s[num + 1:]).lower():
        print("AC")
    else:
        print("WA")


def abc104_c():
    import itertools
    d, g = map(int, input().split())
    p = [0] * d
    c = [0] * d
    ans = 1000
    for i in range(d):
        p[i], c[i] = map(int, input().split())

    for i in range(d + 1):
        for seq in itertools.combinations([i for i in range(d)], i):
            print(seq)
    print(ans)


# abc104_c()


def abc104_d():
    n = int(input())
    print(n)


# abc104_d()


def abc105_a():
    n, k = map(int, input().split())

    print([0, 1][n % k != 0])


def abc105_b():
    n = int(input())
    for i in range(15):
        for j in range(26):
            if n == (i * 7 + j * 4):
                print("Yes")
    print("No")


def abc105_c():
    n = int(input())
    if not n:
        print(0)
    elif n > 0:
        n_bin = str(bin(n)[2:])
        digit = len(n_bin)
        for i in range(len(n_bin) - 1):
            if i % 2:  # -2, -8...
                if n_bin[digit - i - 1] == "2":
                    n_bin = n_bin[:digit - i - 2] + str(int(n_bin[digit - i - 2]) + 1) + "0" + n_bin[digit - i:]
                elif n_bin[digit - i - 1] == "1":
                    n_bin = n_bin[:digit - i - 2] + str(int(n_bin[digit - i - 2]) + 1) + n_bin[digit - i - 1:]
            else:
                if n_bin[digit - i - 1] == "2":
                    n_bin = n_bin[:digit - i - 2] + str(int(n_bin[digit - i - 2]) + 1) + "0" + n_bin[digit - i:]
        if n_bin[0] == "2" and len(n_bin) % 2:
            n_bin = "110" + n_bin[1:]

        print(n_bin)
    else:
        print(0)


# abc105_c()


def abc105_d():
    print(0)


# print(abc105_d())


def abc106_a():
    a, b = map(int, input().split())

    print(a * b - a - b + 1)


def abc106_b():
    n = int(input())
    count = 0
    for i in range(1, n + 1, 2):
        div = 0
        for j in range(1, i + 1, 2):
            if i % j == 0:
                div += 1
        if div == 8:
            count += 1

    print(count)


def abc106_c():
    s = input()
    k = int(input())

    if s[:k].replace("1", "") == "":
        print(1)
    else:
        print((s.replace("1", ""))[0])


def abc106_d():
    import numpy as np

    n, m, q = map(int, input().split())
    ls = np.empty(m, dtype=int)
    rs = np.empty(m, dtype=int)
    ps = np.empty((q, 1), dtype=int)
    qs = np.empty((q, 1), dtype=int)
    for i in range(m):
        ls[i], rs[i] = map(int, input().split())
    for i in range(q):
        ps[i], qs[i] = map(int, input().split())

    for i in range(q):
        print(np.sum((ls >= ps[i])[rs <= qs[i]]))
    # for count in np.sum((ls >= ps) * (rs <= qs), axis=1):
    #     print(count)
    print(0)


# abc106_d()


def abc107_a():
    n, i = map(int, input().split())
    print(n - i + 1)


def abc107_b():
    import numpy as np

    h, w = map(int, input().split())
    a = []
    for i in range(h):
        tmp = list(input())
        if not tmp == ["."] * w:
            a = np.append(a, tmp)
    a = a.reshape((-1, w))
    for row in a[:, np.any(a != np.array(["."] * w), axis=0)]:
        print("".join(row))


def abc107_c():
    import numpy as np

    n, k = map(int, input().split())
    x = np.array(list(map(int, input().split())))
    if x[0] >= 0:
        print(x[k - 1])
    elif x[-1] <= 0:
        print(x[n - k])
    else:
        t = np.sum(x < 0)
        if x[t] == 0:
            k = k - 1
            x = np.delete(x, [t])

        time = np.inf
        if t >= k:
            time = x[t - k]
        if (n - t) >= k:
            time = min(time, x[t + k - 1])

        for i in range(max(1, k - (n - t)), min(t + 1, k)):
            left = -x[t - i]
            right = x[t + k - i - 1]

            tmp = min(left, right) * 2 + max(left, right)
            if tmp < time:
                time = tmp

        print(time)


def abc107_d():
    n = int(input())
    a, b = map(int, input().split())

    print(n, a, b)


# print(abc107_d())


def abc108_a():
    k = int(input())
    if k % 2 == 0:
        print(int(k ** 2 / 4))
    else:
        print(int((k // 2) * (k // 2 + 1)))


def abc108_b():
    x1, y1, x2, y2 = map(int, input().split())

    x3 = x2 - (y2 - y1)
    y3 = y2 + (x2 - x1)
    x4 = x3 - (y3 - y2)
    y4 = y3 + (x3 - x2)

    print("{} {} {} {}".format(x3, y3, x4, y4))


def abc108_c():
    import numpy as np

    n, k = map(int, input().split())

    count = 0
    ks = np.array([i * k for i in range(1, (3 * n) // k + 1)])

    for a in range(1, n + 1):
        tmp = ks - a
        b_candidate = tmp[(1 <= tmp) & (tmp <= n)]
        c_candidate = b_candidate
        for b in b_candidate:
            count += np.sum((b + c_candidate) % k == 0)

    print(count)


# print(abc108_c())


def abc108_d():
    n, i = map(int, input().split())

    print(n, i)


def abc109_a():
    a, b = map(int, input().split())
    print(["No", "Yes"][a * b % 2])


def abc109_b():
    n = int(input())
    words = [input() for _ in range(n)]
    for i in range(n - 1):
        if words[i][-1] != words[i + 1][0]:
            print("No")
    if len(set(words)) != len(words):
        print("No")
    print("Yes")


def abc109_c():
    import numpy as np
    import functools
    import math
    n, x = map(int, input().split())
    x = np.abs(np.array(list(map(int, input().split()))) - x)
    gcd = functools.reduce(math.gcd, x)
    print(gcd)


def abc109_d():
    n = int(input())
    print(n)


# print(abc109_d())

def abc111_a():
    n = int(input())
    print(1110 - n)


def abc111_b():
    n = int(input())
    for i in range(1, 10):
        if 111 * i >= n:
            print(111 * i)


def abc111_c():
    import collections

    b_len = int(int(input()) / 2)
    a = list(map(int, input().split()))
    b1 = a[::2]
    b2 = a[1::2]
    b1_count = collections.Counter(b1).most_common()
    b2_count = collections.Counter(b2).most_common()

    if len(b1_count) == 1:
        if len(b2_count) == 1:
            if b1_count[0][0] == b2_count[0][0]:
                print(b_len)
            else:
                print(0)
        else:
            if b1_count[0][0] == b2_count[0][0]:
                print(b_len - b2_count[1][1])
            else:
                print(b_len - b2_count[0][1])
    elif len(b2_count) == 1:
        if b2_count[0][0] == b1_count[0][0]:
            print(b_len - b1_count[1][1])
        else:
            print(b_len - b1_count[0][1])
    else:
        if b1_count[0][0] == b2_count[0][0]:
            print(2 * b_len - max(b1_count[0][1] + b2_count[1][1], b1_count[1][1] + b2_count[0][1]))
        else:
            print(2 * b_len - b1_count[0][1] - b2_count[0][1])


def abc111_d():
    n = int(input())
    print(n)


# print(abc111_d())

def abc112_a():
    n = int(input())
    if n == 1:
        print("Hello World")
    else:
        a = int(input())
        b = int(input())
        print(a + b)


def abc112_b():
    n, t_lim = map(int, input().split())
    c = []
    for _ in range(n):
        ci, t = map(int, input().split())
        if t <= t_lim:
            c.append(ci)
    if c:
        print(min(c))
    else:
        print("TLE")


def abc112_c():
    n = int(input())
    x_list = [0] * n
    y_list = [0] * n
    h_list = [0] * n
    for i in range(n):
        x_list[i], y_list[i], h_list[i] = map(int, input().split())

    h_max = max(h_list)
    h_argmax = h_list.index(h_max)
    for cx in range(101):
        for cy in range(101):
            flag = False
            if h_max > 0:
                h = h_max + abs(x_list[h_argmax] - cx) + abs(y_list[h_argmax] - cy)
                for i in range(n):
                    if max(h - abs(x_list[i] - cx) - abs(y_list[i] - cy), 0) != h_list[i]:
                        flag = True
                        break
                if flag:
                    continue
                else:
                    print("{} {} {}".format(cx, cy, h))
            else:
                pass

    # print("{} {} {}".format(cx, cy, h)


# print(abc112_c())


def abc112_d():
    n = int(input())
    print(n)


# print(abc112_d())


def tpbc_2018_a():
    s = input()
    if len(s) == 2:
        print(s)
    else:
        print(s[::-1])


def tpbc_2018_b():
    a, b, k = map(int, input().split())
    for i in range(k // 2):
        a //= 2
        b += a
        b //= 2
        a += b
    if k % 2 == 1:
        a //= 2
        b += a
    print("{} {}".format(a, b))


def tpbc_2018_c():
    n = int(input())
    a = [int(input()) for _ in range(n)]
    a.sort()
    low = a[:len(a) // 2]
    high = a[len(a) // 2:]
    if len(a) % 2 == 0:
        print(2 * (sum(high) - sum(low)) + max(low) - min(high))
    else:
        low2 = a[:len(a) // 2 + 1]
        high2 = a[len(a) // 2 + 1:]
        print(max(2 * (sum(high) - sum(low)) - high[0] - high[1], 2 * (sum(high2) - sum(low2)) + low2[-2] + low2[-1]))


def tpbc_2018_d():
    n = int(input())
    print(n)


# print(tpbc_2018_d())


def abc113_a():
    x, y = map(int, input().split())
    print(int(x + y / 2))


def abc113_b():
    import numpy as np
    _ = int(input())
    t, a = map(int, input().split())
    h = np.array(list(map(int, input().split())))
    temp = np.abs(a - (t - h * 0.006))
    print(np.argmin(temp) + 1)


def abc113_c():
    import numpy as np
    n, m = map(int, input().split())
    city = np.array([list(map(int, input().split())) for _ in range(m)])
    city_sort = [[1000000] for _ in range(m)]
    for i in range(m):
        city_sort[city[i][0] - 1].append(city[i][1])
    print(city_sort)
    # for i in range(m):
    #     city_index = np.argsort(city_sort)
    # print(city_index)
    for i in range(m):
        p = city[i][0]
        x = np.where(city_sort[city[i][0] - 1] == city[i][1])
        print('{0:06d}{0:06d}'.format(p, x))
    print(0)


# abc113_c()


def abc113_d():
    n = int(input())
    print(n)


# print(abc113_d())


def abc114_a():
    x = int(input())
    if x in [3, 5, 7]:
        print("YES")
    else:
        print("NO")


def abc114_b():
    s = input()
    min_diff = 1000
    for i in range(len(s) - 2):
        tmp = abs(753 - int(s[i:i + 3]))
        if tmp < min_diff:
            min_diff = tmp
    print(min_diff)


def abc114_c():
    import numpy as np
    s = input()
    n = int(s)
    ans = 0

    nums = np.array([3, 5, 7])
    tmp = nums.copy()
    for i in range(1, len(s)):
        tmp = (tmp + [[3 * 10 ** i], [5 * 10 ** i], [7 * 10 ** i]]).flatten()
        nums = np.append(nums, tmp)

    for val in nums:
        str_val = str(val)
        if val <= n and "3" in str_val and "5" in str_val and "7" in str_val:
            ans += 1

    print(ans)


def abc114_d():
    def calc_divisor(_n):
        div = [1, _n]
        for _i in range(2, _n // 2 + 1):
            if _n % _i == 0:
                div.append(i)
        print(sorted(div))

    n = int(input())
    divisor = [1]
    for i in range(2, n + 1):
        divisor.extend(calc_divisor(i)[1:])

    print(divisor)


# print(abc114_d())


def abc115_a():
    d = int(input())

    print("Christmas" + " Eve" * (25 - d))


# abc115_a()


def abc115_b():
    n = int(input())
    p = [int(input()) for _ in range(n)]
    p.sort(reverse=True)

    print(p[0] // 2 + sum(p[1:n]))


# abc115_b()


def abc115_c():
    n, k = map(int, input().split())
    h = [int(input()) for _ in range(n)]
    h.sort()

    diff = h[-1] - h[0]
    for i in range(n - k + 1):
        if h[i + k - 1] - h[i] < diff:
            diff = h[i + k - 1] - h[i]

    print(diff)


# abc115_c()


def abc115_d():
    n, x = map(int, input().split())
    digit = [1] * 51
    p = [1] * 51
    for i in range(50):
        digit[i + 1] = 2 * digit[i] + 3
        p[i + 1] = p[i] * 2 + 1

    ans = 0
    while n:
        if x == digit[n]:
            ans += p[n]
            break
        elif x > (digit[n] // 2):
            ans += p[n - 1] + 1
            x -= (digit[n - 1] + 2)
            n -= 1
        else:
            x -= 1
            n -= 1
    if x == 1:
        ans += 1

    print(ans)


# abc115_d()


def caddi2018b_a():
    n = input()
    print(n.count("2"))


# caddi2018b_a()


def caddi2018b_b():
    n, h, w = map(int, input().split())
    boards = [list(map(int, input().split())) for _ in range(n)]
    ans = 0

    for i in range(n):
        if boards[i][0] >= h and boards[i][1] >= w:
            ans += 1
    print(ans)


# caddi2018b_b()


def caddi2018b_c():
    from collections import Counter

    def prime_decomposition(num):
        i = 2
        table = []
        while i * i <= num:
            while num % i == 0:
                num /= i
                table.append(i)
            i += 1
        if num > 1:
            table.append(int(num))
        return table

    n, p = map(int, input().split())
    ans = 1
    factors = Counter(prime_decomposition(p))
    for val in factors:
        ans *= val ** (factors[val] // n)
    print(ans)


# caddi2018b_c()


def caddi2018b_d():
    n = int(input())
    a = [int(input()) for _ in range(n)]
    for val in a:
        if val % 2 == 1:
            print("first")
            return 0
    print("second")


# caddi2018b_d()


def angle():
    import math
    a, b, c = map(float, "7.68059 15.36119 26.22319".split())
    x1, y1, z1 = map(float, input().split())
    x2, y2, z2 = map(float, input().split())

    print(math.degrees(math.atan2(c * abs(z2 - z1), ((a * (x2 - x1)) ** 2 + (b * (y2 - y1)) ** 2) ** 0.5)))

# angle()
