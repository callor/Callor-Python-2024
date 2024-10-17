num1 = int(input("첫번째 숫자를 입력해 주세요: "))
num2 = int(input("첫번재 보다 큰 숫자를 입력해 주세요: "))

if num1 > num2 :
    num1 = num1 + num2
    num2 = num1 - num2
    num1 = num1 - num2
    
hap = 0
for index in range(num1, num2+1) :
    hap = hap + index
    
print("{} 부터 {} 까지의 합 {}".format(num1, num2, hap))    
