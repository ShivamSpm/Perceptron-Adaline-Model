import perceptron as pt         # Question 1
import adaline as ad            # Question 2
from normalizeData import *     # Question 3
import inspect
from linearAlgebra import *

def question1():

    if (len(inspect.signature(pt.Perceptron).parameters) != 3):
        print("Error in Perceptron creation")

    p = pt.makePerceptron(.2, 2, lambda: .5, 1)
    if (p !=
            pt.Perceptron(eta=0.2, weightColMat=Matrix(3, 1, data=[0.5, 0.5, 0.5]), trace=1)):
        print("Error in makePerceptron")

    if (pt.sigma(0.5) != 1 or pt.sigma(-1) != 0):
        print("Error in sigma function")

    if (pt.applyPerceptronVec(p, Vector([0, 1])) != 1):
        print("Error in apply perceptron")

    binary_train_Once_P = pt.trainOnce(p, Vector([0, 1]), 0)
    if (binary_train_Once_P != True):
        print("Error in Perceptron Train Once")

    if pt.andDataSet != [(Vector([1, 1]), 1), (Vector([1, 0]), 0), (Vector([0, 1]), 0), (Vector([0, 0]), 0)]:
        print("Error in perceprtron preprocess function")

    pt.trainEpoch(p, pt.andDataSet)



def question2():
    if (len(inspect.signature(ad.Adaline).parameters) != 3):
        print("Error in Adaline creation")

    a = ad.makeAdaline(.2, 2, lambda: 0.0, 1)
    if (a != ad.Adaline(eta=0.2, weightColMat=Matrix(3, 1, data=[0.0, 0.0, 0.0]), trace=1)):
        print("Error in makeAdeline fuction")

    if (ad.sigma(0.5) != 1 or ad.sigma(0) != 0 or ad.sigma(-0.5) != -1):
        print("Error in Adeline sigma function")

    if (ad.applyAdalineVec(a, Vector([-1,1])) != 0):
        print("Error in applyAdeline function")

    ad.trainOnce(a, Vector([0, 1]), 0)

    if ad.andDataSet != [(Vector([1,1]),1), (Vector([1,-1]),-1), (Vector([-1,1]),-1), (Vector([-1,-1]),-1)]:
        print("Error in Adeline preprocess function")

    ad.trainEpoch(a, ad.andDataSet)



def question3():
    D = dataSetFromFileName("flowers.csv")
    if (D[0] != (Vector(data=[0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1]), 0) or
        D[-1] != (Vector(data=[0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1]), 1)):
        print("Error in normalizing flowers data")



def main():
    question1()
    question2()
    question3()

if __name__ == '__main__':
    main()