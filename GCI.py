def prime_number():
    import numpy as np
    n = int(input())
    if n == 2 or n == 3:
        print(n)
        return 0

    divnums = [i for i in range(3, n, 2)]


prime_number()
