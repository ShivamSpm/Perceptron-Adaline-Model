import csv
from collections import namedtuple
import perceptron
import adaline
Vector = namedtuple('Vector',['data'])
def dataSetFromFileName(file):
    lst = []
    data1 = []
    data2 = []
    data3 = []
    with open(file) as f:
        reader = csv.reader(f)
        for row in reader:

            del row[1::2]
            # print(row)

            if row[2]=='Iris-setosa' or row[2] =='Iris-versicolor':
                lst.append(row)
                # print(row)
                data1.append(float(row[0]))
                data2.append(float(row[1]))
                data3.append(row[2])

        for index,ele in enumerate(data3):
            if ele == 'Iris-setosa':
                data3[index] = 0
            else:
                data3[index] = 1


        min1,max1 = min(data1), max(data1)
        for i, value in enumerate(data1):
            data1[i] = (value - min1) / (max1 - min1)

        min2, max2 = min(data2), max(data2)
        for i, value in enumerate(data2):
            data2[i] = (value - min2) / (max2 - min2)

        data1 = [element * 100 for element in data1]
        data2 = [element * 100 for element in data2]
        data1 = [int(item) for item in data1]
        data2 = [int(item) for item in data2]

        for j,ele in enumerate(data1):
            data1[j] = bin(ele)[2:].zfill(7)
        for j,ele in enumerate(data2):
            data2[j] = bin(ele)[2:].zfill(7)

        concat = []
        vec = []
        for k in range(len(data1)):
            concat.append(data1[k] + data2[k])
        for k in range(len(concat)):
            ele = [int(d) for d in str(concat[k])]
            vec.append(ele)

        D = []
        for i in range(len(data3)):
            D.append((Vector(vec[i]),data3[i]))
        return D


D = dataSetFromFileName('flowers.csv')
def learnPerceptron(dataset):
    p= perceptron.makePerceptron(.1,14,lambda :.5,1)
    perceptron.train(p,D,40)

def learnAdaline(dataset):

    for i in dataset:
        for j in range(len(i[0].data)):
            if i[0].data[j]==0:
                i[0].data[j] = -1


    a = adaline.makeAdaline(.1, 14, lambda: 0.0, 1)
    adaline.train(a, dataset, 40)


if __name__ == '__main__':
    learnPerceptron(D)
    learnAdaline(D)
