import numpy as np

#first task

def get_bayer_masks(n_rows, n_cols):
    mask = np.zeros([3, n_rows, n_cols])
    for i in range(n_rows):
        for j in range(n_cols):
          a, b = i%2, (j+1)%2
          color = (a or b) + (a and b)
          mask[color][i][j] = 1
          #print(color)
    return mask

#second task

def get_colored_img(raw_img):
    n_rows, n_cols = raw_img.shape[0], raw_img.shape[1]
    col_img = get_bayer_masks(n_rows, n_cols)
    for i in range(n_rows):
      for j in range(n_cols):
        for k in range(3):
          if (col_img[k][i][j]==1):
            col_img[k][i][j] = raw_img[i][j] 
    return col_img

A = np.array([[0,1,2],[4,5,6],[3,8,2]])
print(get_colored_img(A))

#third task

def bilinear_interpolation(colored_img):
    _, n_rows, n_cols = colored_img.shape
    bilinear = np.zeros([3, n_rows-2, n_cols-2])
    for i in range(1,n_rows-1):
      for j in range(1, n_cols-1):
        for t in range(3):
          sum=0
          for k in range(-1,2):
            for l in range(-1,2):
              sum += colored_img[t][i+k][j+l]
      #print(sum)        
          bilinear[t][i-1][j-1] = int(sum / 9) # or /9 
    return bilinear

#fourth task
