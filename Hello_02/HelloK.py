
num = int(input("숫자를 입력해 주세요: "))
for i in range(2, num) :
    if num % i == 0 :
        break

if i == num - 1 :
    print("{}는 소수입니다.".format(num))   
else :
    print("{}는 소수가 아닙니다.".format(num))
    