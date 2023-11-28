# 두개의 서로다른 숫자를 입력받아
# 큰 수를 출력하는 프로그램을 작성하시오

num1 = input("첫번째 숫자를 입력하세요 : ")
num2 = input("두번째 숫자를 입력하세요 : ")
if num1 > num2:
    print("큰 수는 {}이고 작은수는 {} 입니다.".format(num1,num2))
else:
    print("큰 수는 {}이고 작은수는 {} 입니다.".format(num2,num1))
    