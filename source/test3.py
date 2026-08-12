list1 = [1,2,3]
strr = "".join([str(d) for d in list1])

print(strr)
print(int(strr) + 1)

print(list(str(int(strr) + 1)))