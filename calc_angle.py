def angle():
    import math
    a = 7.68059
    b = 15.36119
    c = 26.22319
    x1, y1, z1 = map(float, input("x1 y1 z1: ").split())
    x2, y2, z2 = map(float, input("x2 y2 z2: ").split())

    print("2D:")
    print(math.degrees(math.atan2(c * abs(z2 - z1), b * abs(y2 - y1))))
    print("3D:")
    print(math.degrees(math.atan2(c * abs(z2 - z1), ((a * (x2 - x1)) ** 2 + (b * (y2 - y1)) ** 2) ** 0.5)))


angle()
