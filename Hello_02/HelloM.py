
subject = 1
total = 0
while subject <= 5 :
    score = int(input("{} 번의 점수를 입력하세요: ".format(subject)))
    
    if score < 0 or score > 100 :
        print("유효한 성적이 아닙니다. 다시 입력해주세요.")
        continue    
    
    print("{} 번의 점수 : {} ".format(subject,score))
    total = total + score
    subject = subject + 1
    
print("총점은 {}점이고, 평균은 {}점 입니다.".format(total,total/5))    