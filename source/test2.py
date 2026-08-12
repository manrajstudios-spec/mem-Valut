
nums = [1,1,1,3]
k = 2

l = 0
r = 0

max_freq = {}
max_sum = 1

cur_freq = {nums[l]:1}
cur_sum = 1

while True:
    if l+r+1<len(nums):
        cur_r = nums[l+r+1]

        found_r_freq = cur_freq.get(cur_r,0)
        
        if found_r_freq:
            if found_r_freq < k:
                cur_freq[cur_r] += 1
                cur_sum += 1
                r += 1
            else:
                cur_freq[nums[l]] -= 1
                l += 1
                cur_sum -= 1
                r-=1 
                
                if l==r:
                    break
        else:
            cur_freq[cur_r] = 1
            cur_sum += 1
            r += 1 
    else:
        break
    
    print(cur_freq,cur_sum)
    if cur_sum > max_sum:
        max_sum = cur_sum

if cur_sum > max_sum:
    max_sum = cur_sum 

print(max_sum)