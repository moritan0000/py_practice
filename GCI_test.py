A = {'t': 64, 'j': 53, 'q': 47, 'h': 58, 'y': 64, 'a': 75, 'f': 65}


def test1(A):
    print(A)
    return A


A = [11, 9, 27, 29, 31, 19, 16, 24, 6, 5, 33, 12, 18, 4, 14, 10, 39, 35, 23, 26, 1, 34, 8, 20, 21, 17, 2, 38, 15, 3, 13,
     32, 22, 7, 37, 36, 25, 28, 30]
B = ['b', 's', 't', 'e', 'w', 'w', 'j', 'q', 'e', 'w', 't', 'x', 'd', 'z', 'd', 'o', 'g', 'q', 'i', 'g', 's', 'o', 'p',
     'w', 'p', 'b', 'o', 'a', 'e', 'f', 'z', 's', 'c', 's', 'q', 'j', 'v', 'z', 'w']


def test2(A, B):
    t = list(zip(A, B))
    print(t)
    return t


A = [19, 2, 15, 14, 4, 13, 23, 1, 10, 22, 7, 5, 18, 12, 21, 17, 20, 6, 16, 9]


def test3(A):
    A.sort()
    print(A)
    return A


A = ['29', '16', '10', '6', '2', '12', '26', '15', '14', '18', '17', '4', '25', '19', '5', '13', '20', '11', '8', '22',
     '21', '9', '27', '7', '28', '24', '23', '1', '3']


def test4(A):
    A = ".".join(A)
    print(A)
    return A


A = [21, 14, 9, 13, 15, 16, 23, 5, 3, 1, 19, 20, 8, 17, 24, 6, 22, 12, 4, 18, 2, 7, 10, 11]


def test5(A):
    A = [i % 2 for i in A]
    print(A)
    return A


A = (
    'v', 'k', 'b', 'f', 'p', 't', 'a', 'f', 'r', 's', 'u', 's', 'w', 'm', 'e', 'a', 'x', 'v', 'w', 'q', 'o', 'i', 'z',
    'h',
    'q', 'r', 'h', 'v', 'a', 'a', 'x', 'u', 'f', 'y', 'e', 'a', 'e', 'v', 'p', 'p', 'v', 'z', 'v', 'd', 'y', 'o', 'x',
    'e',
    'c', 'y', 'l', 'v', 'j', 'p', 'q', 'j', 'y', 'k', 'm', 'y', 's', 'u', 'v', 'k', 'z', 'j', 'v', 't', 'w', 'o', 'r',
    's',
    'g', 'h', 'h', 'm', 'i', 'l', 'y', 'u', 'z', 'q', 'l', 'y', 't', 'l', 'g', 'x', 'k', 'j', 'v', 'c', 'y', 'k', 'e',
    'a',
    'l', 'n', 'q', 'h', 'r')
B = "n"


def test6(A, B):
    c = A.count(B)
    print(c)
    return c


A = [16, 35, 11, 7, 29, 32, 27, 31, 10, 17, 21, 9, 22, 12, 24, 14, 4, 30, 18, 34, 33, 26, 8, 1, 5, 2, 15, 25, 23, 13,
     28, 6, 19, 3, 20]


def test7(A):
    A = [i for i in A if i % 2 == 0]
    print(A)
    return A


A = 8504.398290082983


def test8(A):
    A = "{:.5f}".format(A)
    print(A)
    return A


A = [[4, 2, 3, 6, 7, 1, 8, 5], [6, 4, 3, 8, 2, 5, 7, 1]]


def test9(A):
    return [list(i) for i in zip(A[0], A[1])]


A = [24, 27, 21, 23, 4, 2, 19, 25, 29, 17, 6, 14, 16, 30, 11, 1, 31, 20, 12, 33, 10, 32, 22, 26, 3, 18, 15, 13, 7,
     28, 8, 5, 9]


def test10(A):
    A = A[::-1]
    print(A)
    return A
