num = int(input("숫자를 입력해 주세요: ")) 

hap = 0
for index in range(1,num + 1,2) :
    print(index,  end=", "  )
    hap = hap + index
    
print("\n{} 까지의 홀수의 합 {}".format(num, hap))