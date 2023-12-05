num = int(input("배수를 구할 숫자를 입력해 주세요: "))

i = 1
hap = 0
while i <= 1000 :
    if i % num == 0 :
        print("{}의 배수: {}".format(num, i))
        hap = hap + i
    i = i + 1
    
print("{}의 배수의 합 {}".format(num, hap))    