def _if(a, b):
    # long
    if a == 10 and b == 10:
        print('both')
    elif a != 10 and b != 10:
        print('neither')
    else:
        print('either')

    # short
    print(['neither', 'either', 'both'][(a == 10) + (b == 10)])

    # long
    if a < b:
        c = 4
    else:
        c = 2

    # short
    c = 4 if a < b else 2
    c = [2, 4][a < b]
    c = 2 + 2 * (a < b)

    # long
    t = a % 3
    if t == 0:
        a = a * 2
    elif t == 1:
        a = 0
    elif t == 2:
        a = t * 2

    # short
    a = [a * 2, 0, (a % 3) * 2][a % 3]

    # short
    t = a % 3
    a = [a * 2, 0, t * 2][t]


def _input():
    # long
    a = int(input())
    b = int(input())
    c = int(input())

    # short
    i = input
    a = int(i())
    b = int(i())
    c = int(i())

    # short
    j = lambda: int(input())
    a = j()
    b = j()
    c = j()

    # long
    a, b = map(int, input().split())
    x, y = map(int, input().split())

    # short
    a, b, x, y = map(int, (input() + " " + input()).split())

    # long
    t = list(map(int, input().split(' ')))
    print('OK') if t[3] or t[0] + t[1] + t[2] < 2 else print('NG')

    # short
    *a, b = map(int, input().split())
    print(['NG', 'OK'][sum(a) < 2 or b])

    # tip
    a = [1, 2, 3, 4]
    *b, c = a  # -> b = [1, 2, 3], c = 4
    *b, c, d = a  # -> b = [1, 2], c = 3, d = 4
    b, *c = a  # -> b = 1, c = [2, 3, 4]
    b, *c, d = a  # -> b = 1, c = [2, 3], d = 4


def _range():
    # long
    for i in range(10):
        pass

    # short when if... does not contain i
    for i in "1" * 10:
        pass


def _math(n):
    # long
    from math import sqrt
    m = sqrt(n)

    # short
    m = n ** 0.5

    # long
    def hoge(i):
        return i ** 2

    # short
    m = lambda i: i ** 2

    # long
    m = n <= 3

    # short
    m = n < 4

    # long
    n * (n - 1) * 5

    # short
    n * ~-n * 5


def _bool_precise():
    # long
    b = False

    # short
    b = 0 > 1

    # [long, short]
    return [True, 1 > 0]


def _string_check(a, b, c, d):
    # long
    n = a[0] == b[0] and a[-1] == b[-1]

    # short
    n = a[0] + a[-1] == b[0] + b[-1]

    # a と b, c と d の長さが同じ(または結合しても一意である)場合
    # long
    n = a == b and c == d

    # short
    n = a + c == b + d


def _recursive_func():
    # long
    def LongLongHoge(a):
        if a < 3:
            return a
        return LongLongHoge(a - 2) * LongLongHoge(a - 1)

    # short
    def LongLongHoge2(a):
        if a < 3:
            return a
        return t(a - 2) * t(a - 1)

    t = LongLongHoge2

    # short
    LongLongHoge = t = lambda a: a if a < 3 else t(a - 2) * t(a - 1)


def _rotate_to_right():
    M = [[0 for i in range(100)] for i in range(100)]
    M = [x[::-1] for x in zip(*M)]
