import numpy as np

def ms_error(data, W, b):
  sum = 0.0
  for i in range(0, len(data)):  # walk thru each item
    X = data[i, 0:2]
    y = data[i, 2]
    z = 0.0
    for j in range(0, len(X)):
      z += X[j] * W[j]
    z += b
    p = 1.0 / (1.0 + np.exp(-z))  # computed result
    sum += (p - y) * (p - y)
  return sum / len(data)  # mean squared error    

def accuracy(data, W, b):
  num_correct = 0
  num_wrong = 0
  for i in range(0, len(data)):  # walk thru each item
    X = data[i, 0:2]
    y = data[i, 2]  # target
    z = 0.0
    for j in range(0, len(X)):
      z += X[j] * W[j]
    z += b
    p = 1.0 / (1.0 + np.exp(-z))  # computed result
    if p > 0.5 and y == 1 or p <= 0.5 and y == 0:
      num_correct += 1
    else:
      num_wrong += 1
  return (num_correct * 1.0) / (num_wrong + num_correct)

def pred_probs(data, W, b):
  # return predicted probabilities in an np.array
  result = np.zeros(shape=(len(data)), dtype=np.float32)
  for i in range(0, len(data)):  # walk thru each item
    X = data[i, 0:2]  # get input values
    z = 0.0
    for j in range(0, len(X)):
      z += X[j] * W[j]
    z += b
    p = 1.0 / (1.0 + np.exp(-z))  # computed result
    result[i] = p
  return result  

def pred_y(pred_probs):
  # return predicted classes in a list 
  result = []
  for i in range(0, len(pred_probs)):  # walk thru probs
    if pred_probs[i] > 0.5:
      result.insert(i, 1)
    else:
      result.insert(i, 0)
  return result  
  
def main():
  print("\nBegin logistic regression with raw Python demo \n")

  np.random.seed(0)

  train_data = np.empty(shape=(6,3), dtype=np.float32)
  train_data[0] = np.array([1.5, 2.5, 1])  # 1
  train_data[1] = np.array([3.5, 4.5, 1])  # 1
  train_data[2] = np.array([6.5, 6.5, 1])  # 1
  train_data[3] = np.array([4.5, 1.5, 0])  # 0
  train_data[4] = np.array([5.5, 3.5, 0])  # 0
  train_data[5] = np.array([7.5, 5.5, 0])  # 0

  print("Training data: \n")
  print(train_data)

  W = np.random.uniform(low = -0.01, high=0.01, size=2)
  b = np.random.uniform(low = -0.01, high=0.01)

  # train
  lr = 0.01
  max_iterations = 70
  indices = np.arange(len(train_data))  # 0,1,2,3,4,5

  print("\nStart training, %d iterations, LR = %0.3f " % (max_iterations, lr))
  for iter in range(0, max_iterations):  # each iteration
    np.random.shuffle(indices)
    for i in indices:  # each training item 
      X = train_data[i, 0:2]  # inputs
      z = 0.0
      for j in range(len(X)):
        z += W[j] * X[j]
      z += b

      p = 1.0 / (1.0 + np.exp(-z))  # computed result
      y = train_data[i, 2]  # target (0 or 1)
     
      # update all weights after each train item
      for j in range(0, 2):  # gradient ascent log likelihood
        W[j] += lr * X[j] * (y - p)  # t - o gives an "add"
      b += lr * (y - p)  # update bias

    if iter % 10 == 0 and iter > 0:
      err = ms_error(train_data, W, b)
      print("epoch " + str(iter) + " Mean Squared Error = %0.4f " % err)

  print("\nTraining complete \n")

  print("Model weights: ")
  print(W)
  print("Model bias:")
  print(b)
  print("")  

  acc = accuracy(train_data, W, b)
  print("Model accuracy on train data = %0.4f " % acc)

  pp = pred_probs(train_data, W, b)
  np.set_printoptions(precision=4)
  print("\nPredicted probabilities: ")
  print(pp)

  preds = pred_y(pp)
  actuals = [1 if train_data[i,2] == 1 else 0 for i in range(len(train_data))]

  print("\nTrain data predicted and actual classes:")
  print("Predicted: ", preds)
  print("Actual   : ", actuals)

  print("\nEnd demo ")

if __name__ == "__main__":
  main()
