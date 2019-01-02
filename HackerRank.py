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


def list_comprehensions():
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input())
    ans = []
    for i in range(x + 1):
        for j in range(y + 1):
            for k in range(z + 1):
                if (i + j + k) != n:
                    ans.append([i, j, k])
    print(ans)


# list_comprehensions()

def nested_list():
    students = {}
    ans = []
    for _ in range(int(input())):
        name = input()
        score = float(input())
        students[name] = score

    target_score = list(set(students.values()))
    target_score.sort()
    for v in students:
        if students[v] == target_score[1]:
            ans.append(v)
    for v in sorted(ans):
        print(v)


def finding_the_percentage():
    n = int(input())
    student_marks = {}
    for _ in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        student_marks[name] = scores
    query_name = input()
    print("{:.2f}".format(sum(student_marks[query_name]) / 3))


# finding_the_percentage()


def itertools_product():
    from itertools import product

    a = list(map(int, input().split()))
    b = list(map(int, input().split()))
    axb = list(product(a, b))
    print(axb[0], end="")
    if len(axb) >= 1:
        for v in axb[1:]:
            print(" ", v, sep="", end="")
        print()


# itertools_product()


def solve(meal_cost, tip_percent, tax_percent):
    from decimal import Decimal
    total_cost = Decimal(meal_cost * (100 + tip_percent + tax_percent) / 100)
    print(total_cost.quantize(Decimal("0")))


# solve(12.00, 20, 8)


def find_second_maximum_number_in_a_list():
    _ = int(input())
    arr = map(int, input().split())
    arr = sorted(list(arr), reverse=True)
    for v in arr[1:]:
        if v != arr[0]:
            print(v)
            break


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


def python_list():
    n = int(input())
    commands = [input().split() for _ in range(n)]
    ans = []
    for c in commands:
        if c[0] == "insert":
            ans.insert(int(c[1]), int(c[2]))
        elif c[0] == "print":
            print(ans)
        elif c[0] == "remove":
            ans.remove(int(c[1]))
        elif c[0] == "append":
            ans.append(int(c[1]))
        elif c[0] == "sort":
            ans.sort()
        elif c[0] == "pop":
            ans.pop()
        elif c[0] == "reverse":
            ans.reverse()


# python_list()


def python_tuples():
    _ = int(input())
    integer_list = map(int, input().split())
    t = tuple(integer_list)
    print(hash(t))


def swap_case(s):
    ans = ""
    for let in s:
        if let.lower() == let:
            ans += let.upper()
        else:
            ans += let.lower()
    return ans


# print(swap_case("Abc"))

def mutate_string(string, position, character):
    return string[:position] + character + string[position + 1:]


def string_validators():
    s = input()
    alphanum = False
    alpha = False
    digit = False
    lower = False
    upper = False
    for let in s:
        if let.isalnum():
            alphanum = True
        if let.isalpha():
            alpha = True
        if let.isdigit():
            digit = True
        if let.islower():
            lower = True
        if let.isupper():
            upper = True
    for b in [alphanum, alpha, digit, lower, upper]:
        print(b)


# string_validators()


def count_substring(string, sub_string):
    ans = 0
    ls = len(sub_string)
    for i in range(len(string) - ls + 1):
        if string[i:i + ls] == sub_string:
            ans += 1
    return ans


def wrap(string, max_width):
    import textwrap
    s_wrap = textwrap.wrap(string, max_width)
    return "\n".join(s_wrap)


def split_and_join(line):
    return "-".join(line.split())


def print_full_name(a, b):
    print("Hello {} {}! You just delved into python.".format(a, b))


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
