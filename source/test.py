import numpy as np

np.random.seed(42)
correct_id = 0

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

relu_output = np.maximum(0,final)

full_pooled = []

for patch in relu_output:
    pooled = []
    for row in range(0,relu_output.shape[1],2):
        cur_row_pool = []
        for col in range(0,relu_output.shape[2],2):
            pool = patch[row:row+2,col:col+2]
            
            cur_row_pool.append(np.max(pool))
            
        pooled.append(cur_row_pool)
        
    full_pooled.append(pooled)
    
full_pooled = np.array(full_pooled)

org_shape = full_pooled.shape

full_pooled = full_pooled.flatten()

# weights and bias from full polled.shape[0] to 2 then softmax

w = np.random.randn(full_pooled.shape[0],2) * 0.01
b = np.zeros((2))

logits = full_pooled @ w + b

e = np.exp(logits - logits.max())

probs = e/e.sum()

loss = -np.log(probs[correct_id])


d_logits = probs.copy()

d_logits[correct_id] -= 1

d_fpooled = d_logits @ w.T 
d_w = full_pooled.T @ d_logits
d_b = d_logits.sum()

d_fpooled = d_fpooled.reshape(org_shape)

print(d_fpooled.shape)

d_relu_out = []

for df in d_fpooled:
    df = df.flatten()
    cur = []
    for row in range(0,relu_output.shape[0],2):
        for col in range(0,relu_output.shape[1],2):
            patch = relu_output[row:row+2,col:col+2]
            
            maxx = patch.max()
            
            print(np.where(patch == patch.max(),df,0))
            break