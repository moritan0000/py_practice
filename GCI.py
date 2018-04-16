def prime_number(n):
    import numpy as np
    nums = np.array([2, 3, 5, 7] + [i for i in range(11, n, 2) if n >= 11 and (i % 3) * (i % 5) * (i % 7) > 0])
    ans = np.array([],dtype=int)

    for val in nums:
        mod = val % nums
        if np.sum(mod == 0) == 1:
            ans = np.append(ans, val)

    return ans


n = 1000
print(prime_number(n))
