# 과목의 점수를 입력받아
# 90점 이상이면 A
# 80점 이상이면 B
# 70점 이상이면 C
# 60점 이상이면 D
# 60점 미만이면 F
# 를 출력하는 프로그램을 작성하시오.

score = int(input("점수를 입력하세요: "))
if score >= 90:
    print("A 학점입니다")
elif score >= 80:
    print("B 학점입니다")
elif score >= 70:
    print("C 학점입니다")
elif score >= 60:
    print("D 학점입니다")
else:
    print("F 학점입니다")    