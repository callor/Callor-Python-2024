hap = 0
count = 0

while True :
    num = int(input("숫자를 입력해 주세요: "))
    hap += num
    count += 1
    if hap >= 1000 :
        break
    
print("1000넘은 수 : {} ".format(hap))
print("평균은 {} 입니다.".format(hap/count))