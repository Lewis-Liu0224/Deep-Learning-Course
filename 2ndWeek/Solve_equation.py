import math as m

a = int(input("Please input number a:"))
b = int(input("Please input number b:"))
c = int(input("Please input number c:"))

isSolve = m.pow(b, 2) - 4 * a * c

if (a == 0):
    print("这不是一个二元一次方程")
elif (isSolve < 0):
    print("次方程无解")
elif (isSolve == 0):
    ans = (-b) / 2 * a
    print("次方程有一解为：")
    print("x = %.3f" % ans)
elif (isSolve > 0):
    ans1 = (-b + m.sqrt(m.pow(b, 2) - 4 * a * c)) / 2 * a
    ans2 = (-b - m.sqrt(m.pow(b, 2) - 4 * a * c)) / 2 * a
    print("次方程有两解分别为：")
    print("x1 = %.3f,x2 =  %.3f" % (ans1, ans2))
