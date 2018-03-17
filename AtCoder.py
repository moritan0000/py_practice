def ac_practice():
    _a = int(input())
    _bc = str(input()).split()
    _b = int(_bc[0])
    _c = int(_bc[1])
    _s = str(input())
    print("{} {}".format(_a + _b + _c, _s))


def agc021_a():
    # 1~inputまでの数字で最大の桁和を出力
    _num = int(input())
    _num2 = _num
    _digit = 0
    # _numの桁数を求める
    while _num2:
        _digit += 1
        _num2 //= 10
    _topdig = _num // 10 ** (_digit - 1)
    # _numが X999...999の場合は_numが桁和最大
    if (_num - _topdig * 10 ** (_digit - 1)) == 10 ** (_digit - 1) - 1:
        print(_topdig + 9 * (_digit - 1))
    # それ以外は(X-1)999...999が桁和最大
    else:
        print(_topdig - 1 + 9 * (_digit - 1))


def abc089_a():
    _num = int(input())
    print(_num // 3)


def abc089_b():
    _num = int(input())
    del _num
    _colors = str(input())
    if _colors.find("Y") == -1:
        print("Three")
    else:
        print("Four")


def abc089_c():
    _num = int(input())
    _namelist = []
    _countM = 0
    _countA = 0
    _countR = 0
    _countC = 0
    _countH = 0

    for i in range(0, _num):
        _name = str(input())
        _namelist.append(_name)
        if _name[0] == "M":
            _countM += 1
        elif _name[0] == "A":
            _countA += 1
        elif _name[0] == "R":
            _countR += 1
        elif _name[0] == "C":
            _countC += 1
        elif _name[0] == "H":
            _countH += 1
    print(_countM * _countA * _countR + \
          _countM * _countA * _countC + \
          _countM * _countA * _countH + \
          _countM * _countR * _countC + \
          _countM * _countR * _countH + \
          _countM * _countC * _countH + \
          _countA * _countR * _countC + \
          _countA * _countR * _countH + \
          _countA * _countC * _countH + \
          _countR * _countC * _countH)


def abc089_d():
    _hwd = str(input()).split()
    _hwd = list(map(int, _hwd))
    _area = []
    _lrs = []
    _nowx = 0
    _nowy = 0

    for i in range(0, _hwd[0]):
        _aij = str(input()).split()
        _area.append(list(map(int, _aij)))

    _q = int(input())

    for i in range(0, _q):
        _lr = str(input()).split()
        _lrs.append(list(map(int, _lr)))

    for i in range(0, _q):
        _mpcount = 0
        for i2 in range(0, int((_lrs[i][1] - _lrs[i][0]) / _hwd[2]) + 1):
            for i3 in range(0, _hwd[0]):
                if _lrs[i][0] + _hwd[2] * i2 in _area[i3]:
                    if not i2:
                        _nowx = _area[i3].index(_lrs[i][0]) + 1
                        _nowy = i3 + 1
                    _nextx = _area[i3].index(_lrs[i][0] + _hwd[2] * i2) + 1
                    _nexty = i3 + 1
            _mpcount += abs(_nextx - _nowx) + abs(_nexty - _nowy)
            _nowx = _nextx
            _nowy = _nexty
        print(_mpcount)


def abc091_a():
    pass




