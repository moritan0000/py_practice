def sock_merchant(n, ar):
    from collections import Counter
    pair = 0
    sock_counter = Counter(ar)
    for v in sock_counter.values():
        pair += v // 2
    return pair


def counting_valleys(n, s):
    now = 0
    valley = 0
    for i in range(n):
        if now == 0 and s[i] == "D":
            valley += 1
        if s[i] == "U":
            now += 1
        else:
            now -= 1
    return valley


def jumping_on_clouds(c):
    from math import ceil
    c_safe = ""
    for v in c:
        c_safe += str(v)
    c_safe = c_safe.split("1")
    step = c.count(1)
    for v in c_safe:
        step += ceil((len(v) - 1) / 2)
    return step


def repeated_string(s, n):
    return s.count("a") * (n // len(s)) + s[:n % len(s)].count("a")


def hourglass_sum(arr):
    total = []
    for i in range(1, 5):
        for j in range(1, 5):
            total.append(sum(arr[i - 1][j - 1:j + 2]) + arr[i][j] + sum(arr[i + 1][j - 1:j + 2]))
    return max(total)


def rot_left(a, d):
    return a[d:] + a[:d]


def minimum_bribes(q):
    num = 0
    for i in range(len(q) - 1):
        if q[i] > i:
            if q[i] - i > 3:
                print("Too chaotic")
                return None
            else:
                num += q[i] - i - 1
        else:
            if q[i] > q[i + 1]:
                num += 1
    print(num)


# minimum_bribes([1, 2, 5, 3, 7, 8, 6, 4])


def simple_array_sum(ar):
    return sum(ar)


def compare_triplets(a, b):
    points = [0, 0]
    for i in range(3):
        if a[i] > b[i]:
            points[0] += 1
        elif a[i] < b[i]:
            points[1] += 1
    return points


def py_if_else():
    n = int(input())
    if n % 2:
        print("Weird")
    else:
        if 6 <= n <= 20:
            print("Weird")
        else:
            print("Not Weird")


def python_arithmetic_operators():
    a = int(input())
    b = int(input())
    print(a + b)
    print(a - b)
    print(a * b)


def python_division():
    a = int(input())
    b = int(input())
    print(a // b)
    print(a / b)


def python_loops():
    n = int(input())
    for i in range(n):
        print(i ** 2)


def is_leap(year):
    leap = False
    if year % 4 == 0 and not (year % 400 and year % 100 == 0):
        leap = True
    return leap


def python_print():
    n = int(input())
    for i in range(1, n + 1):
        print(i, end="")
    print()
