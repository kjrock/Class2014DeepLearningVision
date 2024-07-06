# 1번
s3 = set([i for i in range(3,101,3)])
s5 = set([i for i in range(5,101,5)])
a = s3.intersection(s5)
print(a)


# 2번
a = (5,33, 77)
b = (44,823,11)
c = (10,50,90)

d = [a[i] + b[i] + c[i] for i in range(len(a))]
print(d)

# 3번
class MultiIterator:
    def __init__(self,stop, mul):
        self.current = 0
        self.stop = stop
        self.mul = mul

    def __iter__(self):
        return self

    def __next__(self):
        r = self.current + self.mul
        if r < self.stop:
            self.current += self.mul
            return r
        else:
            raise StopIteration

for i in MultiIterator(20,3):
    print(i)

# 4번
def mutiple_number_generator():
    f = open('words.txt')
    yield from f

for idata in mutiple_number_generator():
    print(idata)

# 5번
import re

emails = ['python@mail.example.com', 'python+kr@example.com',
          'python-dojang@example.co.kr', 'python_10@example.info',
          'python.dojang@e-xample.com', '@example.com',
          'python@example', 'python@example-com']

for email in emails:
    print(True if re.match(r'[A-Za-z0-9\._+-]+@[A-Za-z-]+\.[A-Za-z-.]+', email) else False)

# 6번
import numpy as np

data = np.array([[4,2,7,11,8,80],
                 [9,22,73,41,57,20],
                 [47,29,87,41,33,92],
                 [3,47,44,14,62,80],
                 [34,61,1,51,8,34]])

print('total max:', max(data.max(axis=0)))
print('row sum:', data.sum(axis=0))
print('col ave:', data.mean(axis=1))


# 7번
f = open('UN.txt')
continent = dict()

for data in f:
    data = data.strip()
    contry = data.split(',')
    # print(contry)
    if not continent.get(contry[1]):
        continent[contry[1]] = list()

    continent[contry[1]].append(contry[0])

print('Enter the name of a continent:')
# for name in continent.keys():
#     print(name)

input_data = input()
if continent.get(input_data):
    for name in continent[input_data]:
        print(name)


# 8번
strData = ['good item', 'hello world', 'python programming', 'real data', 'script python']
def find_python():
    is_python = False
    while True:
        s = yield is_python
        is_python = False if s.find('python') == -1 else True

co = find_python()
next(co)
for s in strData:
    print(co.send(s))
