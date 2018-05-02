# Problem No. 1

x = [[1],
     [2],
     [3],
     [4],
     [5]]
y = [[11],
     [12],
     [13],
     [14],
     [15]]

s1 = [["A"],
      ["B"],
      ["C"],
      ["D"],
      ["E"]]
s2 = [["a"],
      ["b"],
      ["c"],
      ["d"],
      ["e"]]


def add_vector(x, y):
    z = x
    for i in range(len(x)):
        z[i][0] += y[i][0]
    return z

print(add_vector(s1, s2))
