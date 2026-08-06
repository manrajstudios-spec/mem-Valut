import numpy as np

array = np.array([[1,2,3,4],
                  [5,6,7,8],
                  [10,11,12,13],
                  [14,15,16,17]])

pooled  =[]
for row in range(0,array.shape[0],2):
    cur_row = []
    for col in range(0,array.shape[1],2):
        pool = array[row:row+2,col:col+2]
        cur_row.append(np.max(pool))
    pooled.append(cur_row)
    
pooled = np.array(pooled)

org_shape = pooled.shape
pooled = pooled.flatten()    

w = np.random.randn(pooled.shape[0],2) * 0.01
b = np.zeros((2))

logits = pooled @ w + b

e = np.exp(logits-logits.max())

probs = e/e.sum()

loss = -np.log(probs[0])

d_logits = probs.copy()

d_logits[0] -= 1

d_pooled = d_logits @ w.T 

d_pooled = d_pooled.reshape(org_shape)

d_relu = []

for dp in d_pooled:
    cur = []
    for row in range(0,array.shape[0],2):
        i = 0
        cur_row = []
        for col in range(0,array.shape[1],2):
            d = dp[i]
            i +=1
            
            patch = array[row:row+2,col:col+2]
            
            cur_row.append(np.where(patch == patch.max(),d,0))
        cur.append(cur_row)  
          
    d_relu.append(np.array(cur))

print(np.array(d_relu).shape)