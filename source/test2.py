import numpy as np

sims = np.array([
    [-np.inf,0.8,0.1,0.1,0.1,0.1,0.1,0.1],
    [0.8,-np.inf,0.8,0.1,0.1,0.1,0.1,0.1],
    [0.1,0.8,-np.inf,0.1,0.1,0.1,0.1,0.1],
    [0.1,0.1,0.1,-np.inf,0.8,0.1,0.1,0.1],
    [0.1,0.1,0.1,0.8,-np.inf,0.8,0.1,0.1],
    [0.1,0.1,0.1,0.1,0.8,-np.inf,0.8,0.1],
    [0.1,0.1,0.1,0.1,0.1,0.8,-np.inf,0.8],
    [0.1,0.1,0.1,0.1,0.1,0.1,0.8,-np.inf]
])

threshold = 0.5
groups = []
last_groups = []

for i,sim in enumerate(sims):
    sim = np.argwhere(sim>=threshold).flatten()
    sim = sim[sim > i].tolist()
    
    sim.append(i)
    
    if not last_groups:
        last_groups.append(set(sim))
        continue    
    
    if last_groups:
        sim = set(sim)
        founded = sim.copy()
        groups = []
        
        for last in last_groups:
            if last & founded:
               founded |= last
            else:
                groups.append(last)
        
        if not founded:
            groups.append(sim)
        else:
            groups.append(founded)  
                  
        last_groups = groups

print(last_groups)