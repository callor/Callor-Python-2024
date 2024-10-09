# 팩토리얼 구하는 코드
# 5! = 5 * 4 * 3 * 2 * 1
# 19! = 121645100408832000

num = int(input("숫자를 입력해 주세요: "))
hap = 1
for i in range(num, 0, -1) :
    hap = hap * i
    
print("{}! = {}".format(num, hap))    
