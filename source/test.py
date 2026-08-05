import numpy as np

image = np.array([[1,2,3,4],
                  [5,6,7,8],
                  [9,10,11,12],
                  [14,15,16,17]])
num_kernels = 2
x_kernel = 3
y_kernel = 3

kernels = np.random.randn(num_kernels,x_kernel,y_kernel)
bias = np.zeros(num_kernels)


final = []

for k in range(num_kernels):
    feature_map = []
    for row in range(image.shape[0] - y_kernel + 1):
        cur_row = []
        for col in range(image.shape[1] - x_kernel +1):
            patch = image[row:row+y_kernel,col:col+x_kernel]
            
            result = np.sum(kernels[k] * patch) + bias[k]
            

            cur_row.append(result)
    
        feature_map.append(cur_row)  
        
    final.append(feature_map)  

final = np.array(final)
print(final)