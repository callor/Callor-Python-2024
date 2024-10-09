
# 2 단부터 9 단까지의 구구단을 출력하는 프로그램을 작성하시오.
# 단, 결과가 짝수인 경우만 출력하시오.

for i in range(2,10) :
    for j in range(1, 10) :
        if (i * j ) % 2 == 0 :
            print("{} x {} = {}".format(i, j, i*j))
    print("\n")