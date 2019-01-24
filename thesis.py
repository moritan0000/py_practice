path = "../../Desktop/"
with open(path + "a.txt") as fin:
    lines = fin.readlines()
for i in range(len(lines)):
    lines[i] = lines[i][:-1] + " "
    if lines[i][0].upper() == lines[i][0]:
        lines[i] = "\n" + lines[i]
with open(path + "a_convert.txt", mode="w") as fout:
    fout.write("".join(lines))
