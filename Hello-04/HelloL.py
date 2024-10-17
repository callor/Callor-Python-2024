print("\n1 ~ 100까지 수 중에서 2의 배수이면서 3의 배수가 아닌수")

i = 0
while i <= 100 :
    i = i + 1
    if (i % 2 == 0) and (i % 3 != 0) :
        print(i, end = " ")