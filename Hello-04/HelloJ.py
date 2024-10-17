
num = int(input("숫자를 입력해 주세요: "))

i = 0
hap = 0
while i <= 100 :
    i = i + 1
    if (i % num) != 0 :
        continue
    hap = hap + i
    
print("{}의 배수 합: {}".format(num, hap))