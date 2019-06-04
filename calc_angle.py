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


# angle()

def describe_data():
    import pandas as pd

    with open("C:/Linux_home/jazz/LAMMPS/Si_slab/MD/Si_slab_4_wo_tag.xyz") as f:
        raw_data = f.readlines()

    force = [[0] for _ in range(len(raw_data))]
    for i in range(len(raw_data)):
        line = raw_data[i].split()
        tmp = []
        tmp.append(line[0])
        tmp.append(int(line[1]))
        tmp.append(line[2])
        tmp.extend(list(map(float, line[3:])))
        force[i] = tmp

    force = pd.DataFrame(force, columns=["file", "no", "atom", "x", "y", "z", "f_x", "f_y", "f_z"])
    force["f_sum"] = force[["f_x", "f_y", "f_z"]].sum(axis=1)
    # print(force)
    print(force.describe())
    force.to_csv("C:/Users/HirotakaMorishita/Desktop/force_4.csv")


describe_data()
