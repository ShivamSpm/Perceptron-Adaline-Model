import itertools
from collections import namedtuple


Adaline = namedtuple('Adaline',['eta','weightColMat','trace'])
Matrix = namedtuple('Matrix',['rows', 'cols', 'data'])
Vector = namedtuple('Vector',['data'])

def makeAdaline(eta,n,func,trace):
    value = func()
    return Adaline(eta,Matrix(n+1,1,list(itertools.repeat(value, n+1))),trace)


def sigma(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1

    else:
        return 0

def applyAdaline(a,augColMat):
    sum = 0
    for i in range(len(a.weightColMat.data)):
        sum += a.weightColMat.data[i]*augColMat.data[i]

    return sigma(sum)



def applyAdalineVec(a,vec):

    sum = 0
    vec.data.append(1)

    for i in range(len(a.weightColMat.data)):
        sum += a.weightColMat.data[i] * vec.data[i]

    return sigma(sum)


def trainOnce(a,inputVec,target):
    sum=0
    inputVec.data.append(1)
    for i in range(len(a.weightColMat.data)):
        sum+= a.weightColMat.data[i] * inputVec.data[i]



    updatedWeight = []
    for i in range(len(a.weightColMat.data)):
        changeInWt = a.eta * (target - sum) * inputVec.data[i]
        updatedWeight.append(changeInWt)

    if any(updatedWeight):
        for k in range(len(a.weightColMat.data)):
            a.weightColMat.data[k] += updatedWeight[k]
        # print(p.weightColMat.data)
        return True
    else:
        return False


v0 = Vector([1, 1])
v1 = Vector([1, -1])
v2 = Vector([-1, 1])
v3 = Vector([-1, -1])

andDataSet = [(v0, 1),
(v1, -1),
(v2, -1),
(v3, -1)]

# ***********************(h)*****************************
orDataSet = [(v0, 1),
(v1, 1),
(v2, 1),
(v3, -1)]

def trainEpoch(a,dataset):
    lst = []
    for inputVec in dataset:
        b = trainOnce(a,inputVec[0],inputVec[1])
        lst.append(b)
    if a.trace==1:
        print("After epoch: ",a)
    if True in lst:
        return True
    else:
        return False

def train(a,dataset,bound):
    value = True
    for _ in range(bound):
        if value:
            trainEpoch(a,dataset)
        else:
            break

if __name__ == '__main__':
    a = makeAdaline(.2, 2, lambda: 0.5, 1)
    train(a, andDataSet, 40)


