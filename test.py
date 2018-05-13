print("Hello, world!")
x = 7
y = 11.0
if 5 < x < y < 20:
    print(x + y)

# comment
list_0 = ['apple',
          'orange',
          'peach',
          'watermelon'
          ]

tuple_0 = 'cat', \
          'dog', \
          'pig', \
          'horse'

dict_0 = {'animal': 'cat',
          'fish': 'shark',
          'bird': 'crow',
          'fruits': 'apple'
          }

print("dict[fruits] = ", dict_0['fruits'])
print("The last element of the list = ", list_0[-1])

list_0.append('cherry')
list_0[2] = 'plum'
print(list_0[::-1])

for element in dict_0:
    print(element)

if 'apple' in list_0:
    print("There's an apple in the list!")


def print_list():
    while True:
        _num = input("Input number 0~4 [q to quit] : ")
        if _num == "q":
            break
        _num = int(_num)
        if not 0 <= _num <= 4:
            continue
        print(list_0[_num].capitalize())


def fizz_buzz():
    for i in range(1, 101):
        if i % 15 == 0:
            print("FizzBuzz")
        elif i % 5 == 0:
            print("Buzz")
        elif i % 3 == 0:
            print("Fizz")
        else:
            print(i)


english = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
japanese = ["月曜日", "火曜日", "水曜日", "木曜日", "金曜日", "土曜日", "日曜日"]
print(dict(zip(english, japanese)))
num_list = [number / 2 for number in range(1, 16) if number % 2 == 1]
print(num_list)
cells = [(row, col)
         for row in range(1, 6) if row % 2 == 1
         for col in range(1, 5) if col % 2 == 0
         ]
for cell in cells:
    print(cell)
word = "abcccddefgghiiiiij"
letter_count = {letter: word.count(letter) for letter in set(word)}
print(letter_count)

number_thing = (number for number in range(1, 12, 2))
print(number_thing)
print(list(number_thing))
print(list(number_thing))


def print_kwargs(*args, **kwargs):
    """
    第1行では引数をタプルにして返す。
    第2行では引数を辞書にして返す。
    """
    print("Arguments:", args)
    print("Keyword arguments:", kwargs)


print_kwargs("Tokyo", "Osaka", "Okinawa", "Hokkaido", "Mikawa-anjo",
             fruits="melon", animal="dog", place="Beijing")
help(print_kwargs)


def run_with_positional_args(func, *args):
    print(func(*args))


def sum_args(*args):
    return sum(args)


run_with_positional_args(sum_args, 27, 581, 2, 21, 445, 1290, 7)


def onikuoishii():
    """
    0~9の数字を1文字ずつランダム生成し、0290141が現れた時点で
    生成された数字を60文字/行ずつ表示し、次行に"お肉おいしい!"と生成数を表示して終了
    マシンパワーを食うので注意
    """
    import random
    num_str = ""
    while True:
        num_str += str(random.randint(0, 9))
        if num_str.endswith("0290141"):
            char_num = 50
            for i in range(0, len(num_str) // char_num + 1):
                print(num_str[i * char_num:(i + 1) * char_num])
            print("お肉おいしい!")
            print("count:", len(num_str))
            break


onikuoishii()

import itertools


def multiply(a, b):
    return a * b


for item in itertools.accumulate([2, 2, 3, 5, 8, 13, 21], multiply):
    print(item)

from pprint import pprint
from collections import OrderedDict

pprint(OrderedDict(dict_0))


class Person():
    def __init__(self, name):
        self.name = name


class MDPerson(Person):
    def __init__(self, name):
        self.name = "Doctor " + name


class EmailPerson(Person):
    def __init__(self, name, email):
        super().__init__(name)
        self.email = email


bob = EmailPerson("Bob", "bob@gmail.com")
dr_bob = MDPerson("Bob")
print(bob.name, bob.email)
print(dr_bob.name)


def print_string():
    _n = 428
    _f = 29.9792458
    _s = "Afternoon tea\u00ea."
    _d = {"n": _n, "f": _f, "s": _s}
    print('{0:<10d}, {2:^20s}, {1:>10f}'.format(_n, _f, _s))
    print("{0[n]}, {0[f]}, {0[s]}, {1}, {2}".format(_d, "twenty", len(_s)))
    print("{_n:d}, {_f:f}, {_s:;>20s}".format(_n=741, _f=123.4567890, _s="Good morning!"))


import re

print_string()

print(re.findall("n.?", "Young Victor Frankenstein"))
print(re.split("n", "Young Victor Frankenstein"))
print(re.sub("n", '?', "Young Victor Frankenstein"))

import string

printable = string.printable
print(printable[:50])
print(printable[50:])
print(re.findall("\s", printable))
source = '''I wish I may, I wish I might
Have a dish of fish tonight.'''
print(re.findall("wish|fish", source))
print(re.findall("[wf]ish", source))

for i in range(1, 256 // 16):
    print(bytes(range((i - 1) * 16, i * 16)))

py_easter = \
    '''
Beautiful is better than ugly.
Explicit is better than implicit.
Simple is better than complex.
Complex is better than complicated.
Flat is better than nested.
Sparse is better than dense.
Readability counts.
Special cases aren't special enough to break the rules.
Although practicality beats purity.
Errors should never pass silently.
Unless explicitly silenced.
In the face of ambiguity, refuse the temptation to guess.
There should be one-- and preferably only one --obvious way to do it.
Although that way may not be obvious at first unless you're Dutch.
Now is better than never.
Although never is often better than *right* now.
If the implementation is hard to explain, it's a bad idea.
If the implementation is easy to explain, it may be a good idea.
Namespaces are one honking great idea -- let's do more of those!
'''
fout = open("py_easter.txt", "wt")
# fout.write(py_easter)
print(py_easter, file=fout, sep="", end="")
fout.close()
print(len(py_easter))

py_easter = ""
with open("py_easter.txt", "rt") as fin:
    chunk = 100
    while True:
        fragment = fin.read(chunk)
        if not fragment:
            break
        py_easter += fragment
    print(len(py_easter))
    while True:
        line = fin.readline()
        if not line:
            break
        py_easter += line
    print(len(py_easter))
    for line in fin:
        py_easter += line
    print(len(py_easter))

bdata = bytes(range(0, 256))
fout = open("bfile.txt", "wb")
fout.write(bdata)
fout.close()

with open("bfile.txt", "rb") as fin:
    print(fin.tell())
    print(fin.seek(255))
    print(fin.seek(-1, 2))

import csv

villians = [
    ['Doctor', 'No'],
    ['Rosa', 'Klebb'],
    ['Mister', 'Big'],
    ['Auric', 'Goldfinger'],
    ['Ernst', 'Blofeld']
]
with open("villians.csv", "wt", newline="") as fout:
    csv_out = csv.writer(fout)
    csv_out.writerows(villians)

with open("villians.csv", "rt") as fin:
    cin = csv.reader(fin)
    villians = [row for row in cin]
    print(villians)

with open("villians.csv", "rt") as fin:
    cin = csv.DictReader(fin, fieldnames=["first", "last"])
    villians = [row for row in cin]
    print(villians)

menu = \
    {
        "breakfast": {
            "hours": "7-11",
            "items": {
                "breakfast burritos": "$6.00",
                "pancakes": "$4.00"
            }
        },
        "lunch": {
            "hours": "11-15",
            "items": {
                "hamburger": "$5.00"
            }
        },
        "dinner": {
            "hours": "15-22",
            "items": {
                "spaghetti": "$8.00"
            }
        }
    }
import json

menu_json = json.dumps(menu)
print(menu_json)
import datetime

now = datetime.datetime.now()
print(now)

import urllib.request as ur

url = "https://raw.githubusercontent.com/koki0702/" \
      "introducing-python/master/dummy_api/fortune_cookie_random1.txt"
conn = ur.urlopen(url)
print(conn.read())
print(conn.status)

import requests

url = "https://raw.githubusercontent.com/koki0702/" \
      "introducing-python/master/dummy_api/fortune_cookie_random2.txt"
resp = requests.get(url)
print(resp)
print(resp.text)
