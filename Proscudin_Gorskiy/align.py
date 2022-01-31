import numpy as np
from PIL import Image
from matplotlib import cm
from numpy import sum, square

def preprocessing(path_img, path_csv):
  pic = Image.open(path_img)
  pix = np.array(pic)
  split_pix= np.array_split(pix, 3, axis=0)
  for i in range(3):
    a,b = np.array(split_pix[i].shape)
    aa,bb=int(a/10), int(b/10)
    step=np.array([a,b,aa,bb])
    split_pix[i] = split_pix[i][aa:a-aa, bb:b-bb]
  csv_data = genfromtxt(path_csv, delimiter=',')
  return split_pix, csv_data, step

  def frame(pix, i, m, n, step):
  a,b,aa,bb = step
  test_pix=split_pix[i][m+aa:m+a-aa, n+bb:n+b-bb]
  return test_pix

def MSE(base, test):
  #minimum
  return sum(square(base-test))/(base.shape[0]*base.shape[1])

def search_min(split_pix, i, a, b, step):
  MSE_min = 10**6
  coord=[0,0] 
  base=frame(split_pix, 1, 0, 0,step)
  for m in range(-a,a):
    for n in range(-b,b):
      test = frame(split_pix, i, m, n, step)
      a = MSE(test, base)
      if (a<MSE_min):
        MSE_min = a
        coord=[m,n]
  return MSE_min, coord

path_img = '/content/gdrive/MyDrive/Colab Notebooks/Proscudin/public_tests/00_test_img_input/img.png'
path_csv='/content/gdrive/MyDrive/Colab Notebooks/Proscudin/public_tests/00_test_img_input/g_coord.csv'
pix, csv, step = preprocessing(path_img, path_csv)

_, point = search_min(split_pix,0,20,20,step)
r = [point[0]+csv[0]-step[0], point[1]+csv[1]]
_, point = search_min(split_pix,2,20,20,step)
b = ([point[0]+csv[0]+step[0], point[1]+csv[1]])
np.array([a,csv,b])