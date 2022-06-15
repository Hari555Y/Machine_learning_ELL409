import argparse
import numpy as np




if __name__ == '__main__':
    args = setup()
    demo(args)


def setup():
    parser = argparse.ArgumentParser()  
    parser.add_argument("--method", default="pinv", help = "type of solver")  
    parser.add_argument("--batch_size", default=5, type=int, help = "batch size")
    parser.add_argument("--lamb", default=0, type=float, help = "regularization constant")
    parser.add_argument("--polynomial", default=10, type=float, help = "degree of polynomial")
    parser.add_argument("--result_dir", default="", type=str, help = "Files to store plots")  
    parser.add_argument("--X", default="", type=str, help = "Read content from the file")
    return parser.parse_args()

def demo(args):
    data = np.genfromtxt(args.X, dtype = str, delimiter = ",")
    if (args.method == "pinv"):
        x = data[:,0]
        x = x.astype(np.float32)
        y = data[:,1]
        y = y.astype(np.float32)
        A = np.zeros((len(x), args.polynomial+1))
        for i in range(len(x)):
            for j in range(args.polynomial+1):
                A[i ,j] = (x[i]**(j))
        b = np.dot(np.dot(np.linalg.inv(np.dot((A.T),A)),(A.T)),y)
        dict_storage = {}
        y_predicted= np.dot(A,b)
        for i in range(len(x)):
            dict_storage[x[i]] = y_predicted[i]
        x.sort()
        y_model =[]
        E = y - y_predicted
        for i in range(len(x)):
            y_model.append(dict_storage[x[i]])

        print(f"weights={y_predicted}")