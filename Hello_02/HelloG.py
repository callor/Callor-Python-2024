print("\n")
num = int(input("출력을 원하는 구구단 단수를 입력하세요: "))
print("\n\n")
print("** {} 단 출력 **".format(num))
for i in range(1,10) :
    print("{} * {} = {}".format(num, i, num*i))
    
print("\n\n")
print("** {} 단 출력 **".format(num))
i = 1
while i < 10 :
    print("{} * {} = {}".format(num, i, num*i))
    i = i + 1    