# predict wheat seed species (0=Kama, 1=Rosa, 2=Canadian)
# from area, perimeter, compactness, length, width,
#   asymmetry, groove

# Anaconda3-2020.02  Python 3.7.6  scikit 0.22.1
# Windows 10/11

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# ---------------------------------------------------------

def show_confusion(cm):
  dim = len(cm)
  mx = np.max(cm)             # largest count in cm
  wid = len(str(mx)) + 1      # width to print
  fmt = "%" + str(wid) + "d"  # like "%3d"
  for i in range(dim):
    print("actual   ", end="")
    print("%3d:" % i, end="")
    for j in range(dim):
      print(fmt % cm[i][j], end="")
    print("")
  print("------------")
  print("predicted    ", end="")
  for j in range(dim):
    print(fmt % j, end="")
  print("")

# ---------------------------------------------------------

def main():
  # 0. prepare
  print("\nBegin Wheat Seeds k-NN using scikit ")
  np.set_printoptions(precision=4, suppress=True)
  np.random.seed(1)

  # 1. load data
  # 0.4410  0.5021  0.5708  0.4865  0.4861  0.1893  0.3452  0
  # 0.4051  0.4463  0.6624  0.3688  0.5011  0.0329  0.2152  0
  # . . .
  # 0.1917  0.2603  0.3630  0.2877  0.2003  0.3304  0.3506  2
  # 0.2049  0.2004  0.8013  0.0980  0.3742  0.2682  0.1531  2

  print("\nLoading train and test data ")
  train_file = ".\\Data\\wheat_train.txt"  # 180 items
  train_X = np.loadtxt(train_file, usecols=[0,1,2,3,4,5,6],
    delimiter="\t", dtype=np.float32, comments="#")
  train_y = np.loadtxt(train_file, usecols=[7],
    delimiter="\t", dtype=np.int64, comments="#")

  test_file = ".\\Data\\wheat_test.txt"  # 30 items
  test_X = np.loadtxt(test_file, usecols=[0,1,2,3,4,5,6],
    delimiter="\t", dtype=np.float32, comments="#")
  test_y = np.loadtxt(test_file, usecols=[7],
    delimiter="\t", dtype=np.int64, comments="#")
  
  print("\nTraining data:")
  print(train_X[0:4])
  print(". . . \n")
  print(train_y[0:4])
  print(". . . ")

  # 2. create and train model
  # KNeighborsClassifier(n_neighbors=5, *, weights='uniform',
  #   algorithm='auto', leaf_size=30, p=2, metric='minkowski',
  #   metric_params=None, n_jobs=None
  # algorithm: 'ball_tree', 'kd_tree', 'brute', 'auto'.

  k = 7
  print("\nCreating kNN model, with k=" + str(k) )
  model = KNeighborsClassifier(n_neighbors=k, algorithm='brute')
  model.fit(train_X, train_y)
  print("Done ")

  # 3. evaluate model
  train_acc = model.score(train_X, train_y)
  test_acc= model.score(test_X, test_y)
  print("\nAccuracy on train data = %0.4f " % train_acc)
  print("Accuracy on test data = %0.4f " % test_acc)

  from sklearn.metrics import confusion_matrix
  y_predicteds = model.predict(test_X)
  cm = confusion_matrix(test_y, y_predicteds)
  print("\nConfusion matrix raw: \n")
  print(cm)
  show_confusion(cm)  # custom formatted

  # 4. use model
  X = np.array([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]],
    dtype=np.float32)
  print("\nPredicting class for: ")
  print(X)
  probs = model.predict_proba(X)
  print("\nPrediction probs: ")
  print(probs)

  predicted = model.predict(X)
  print("\nPredicted class: ")
  print(predicted)

  # 5. TODO: save model using pickle
  import pickle
  print("\nSaving trained kNN model ")
  # path = ".\\Models\\wheat_knn_model.sav"
  # pickle.dump(model, open(path, "wb"))

  # usage:
  # X = np.array([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]],
  #   dtype=np.int64)
  # with open(path, 'rb') as f:
  #   loaded_model = pickle.load(f)
  # pa = loaded_model.predict_proba(x)
  # print(pa)

  print("\nEnd demo ")

if __name__ == "__main__":
  main()
