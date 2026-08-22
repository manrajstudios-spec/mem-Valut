s = "aaabbbccc"
queryCharacters = "abc"
queryIndices = [1, 4, 7]

s = list(s)

all_maxs = []
all_ranges = []
maxx = 0

l,r=0,0

cur_max = 0

while True:
    r += 1
    print(cur_max,l+r)
    if l+r >= len(s):
        cur_range = list(range(l,r))
        l+=1
        r=0
        
        if l >= len(s):break
        
        if cur_max > maxx:
            all_maxs.clear()
            maxx=cur_max
            all_maxs.append(maxx)
            all_ranges.append(cur_range)
        elif cur_max == maxx:
            all_maxs.append(maxx)
            all_ranges.append(cur_range)
            
        cur_max=1
        print("reset")
        continue
    
    if s[l+r] != s[l]:
        cur_range = list(range(l,r))
        l = l+r
        r=0
        
        if cur_max > maxx:
            all_maxs.clear()
            maxx=cur_max
            all_maxs.append(maxx)
            all_ranges.append(cur_range)
        elif cur_max == maxx:
            all_maxs.append(maxx)
            all_ranges.append(cur_range)
        cur_max=1
    else:
        cur_max += 1


cur_range=list(range(l,r+1))

if cur_max > maxx:
    all_maxs.clear()
    maxx=cur_max
    all_maxs.append(maxx)
    all_ranges.append(cur_range)
elif cur_max == maxx:
    all_ranges.append(cur_range)
    all_maxs.append(maxx)

print(all_maxs)