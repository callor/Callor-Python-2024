# 사용자로부터 세 과목의 점수를 입력받아 평균을 구하고
# 평균이 95점 이상이면 "당신의 평균은 00.0점입니다"
# 축하합니다 A+입니다를 출력하고, 
# 마지막에는 평균과 상관 없이
# "감사합니다"를 출력하는 프로그램을 작성하시오


korScore = int(input("국어 점수를 입력하세요 : "))
engScore = int(input("영어 점수를 입력하세요 : "))
mathScore = int(input("수학 점수를 입력하세요 : "))
avg = (korScore + engScore + mathScore) / 3
if avg >= 95:
    print("당신의 평균은 {}점입니다.".format(avg))    
    print("축하합니다 A+입니다")
    
print("감사합니다")


