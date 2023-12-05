num = int(input("숫자를 입력해 주세요: "))

i = 1
hap = 0
while i <= num :
    hap = hap + i
    i = i + 1   
    
print("{} 부터 {} 까지의 합 {}".format(0, num, hap))
print(0, "부터", num, "까지의 합",hap)