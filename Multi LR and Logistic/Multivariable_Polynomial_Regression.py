'''Shreyash Shrivastava
1001397477'''

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

# FIRST ORDER TWO PARAMETER LINEAR REGRESSION
# y = a + b x1 + c x2 + d x1x2

X =[]
Y =[]
Z = []

D = { ((6.4432, 9.6309), 50.9155), ((3.7861, 5.4681) ,29.9852),
((8.1158, 5.2114) ,42.9626), ((5.3283, 2.3159), 24.7445),
((3.5073, 4.8890) ,27.3704), ((9.3900, 6.2406) ,51.1350),
((8.7594, 6.7914), 50.5774), ((5.5016, 3.9552) ,30.5206),
((6.2248, 3.6744) ,31.7380), ((5.8704, 9.8798) ,49.6374),
((2.0774, 0.3774), 10.0634), ((3.0125, 8.8517) ,38.0517),
((4.7092, 9.1329), 43.5320), ((2.3049, 7.9618) ,33.2198),
((8.4431, 0.9871), 31.1220), ((1.9476, 2.6187) ,16.2934),
((2.2592, 3.3536) ,19.3899), ((1.7071, 6.7973) ,28.4807),
((2.2766, 1.3655) ,13.6945), ((4.3570, 7.2123) ,36.9220),
((3.1110, 1.0676) ,14.9160), ((9.2338, 6.5376) ,51.2371),
((4.3021, 4.9417) ,29.8112), ((1.8482, 7.7905) ,32.0336),
((9.0488, 7.1504) ,52.5188), ((9.7975, 9.0372) ,61.6658),
((4.3887, 8.9092) ,42.2733), ((1.1112, 3.3416) ,16.5052),
((2.5806, 6.9875) ,31.3369), ((4.0872, 1.9781) ,19.9475),
((5.9490, 0.3054) ,20.4239), ((2.6221, 7.4407) ,32.6062),
((6.0284, 5.0002) ,35.1676), ((7.1122, 4.7992) ,38.2211),
((2.2175, 9.0472) ,36.4109), ((1.1742, 6.0987) ,25.0108),
((2.9668, 6.1767) ,29.8861), ((3.1878, 8.5944) ,37.9213),
((4.2417, 8.0549) ,38.8327), ((5.0786, 5.7672) ,34.4707) }



for x in D:
    X.append((x[0][0]))
for x in D:
    Y.append((x[0][1]))

for x in D:
    Z.append((x[1]))





# First order regression
# y = a + b x1 + c x2 + d x1x2

# To minimize the cost function and derive the linear constants (a,b,c,d)

mess = input('Enter the order for Multivarible regression calculation: (1,2,3,4)\n')

mess = int(mess)
#mess= 0
if mess == 1:
    def gradient_descent(x1,x2,y):
        x1 = np.array(x1)
        x2 = np.array(x2)
        y = np.array(y)

        # Starting with some value of a,b,c,d and then take steps to minimize the
        # cost function
        a_curr = b_curr = c_curr = d_curr = 0
        iteartions = 10000
        learning_rate = 0.0001
        n = len(x1)

        for i in range(iteartions):
            y_predicted = a_curr + b_curr*x1 + c_curr*x2 + d_curr *x1*x2
            cost = (1/n) * sum([val**2 for val in (y-y_predicted)])

            bd = -(2/n)* sum((y-y_predicted)*x1)
            cd = -(2/n) * sum((y - y_predicted) * x2)
            dd = -(2/n) * sum((y - y_predicted) * x1*x2)
            ad = -(2/n) * sum(y - y_predicted)

            a_curr = a_curr - learning_rate * ad
            b_curr = b_curr - learning_rate * bd
            c_curr = c_curr - learning_rate * cd
            d_curr = d_curr - learning_rate * dd

            #print("a {}, b {}, c {}, d {}, cost {}, iteration {}".format(a_curr,b_curr,c_curr,d_curr,cost,i))

        return [a_curr,b_curr,c_curr,d_curr]

    # Scatter plot 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X, Y, Z, c='r', marker='o')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    x1 = X
    x2 = Y
    y = Z
    # Scatter plot ends

    [a,b,c,d] = gradient_descent(x1,x2,y)

    # Plane plot 3D
    x11 = np.linspace(0,10,10)
    y11 = np.linspace(0,10,10)

    X11,Y11 = np.meshgrid(x11,y11)
    Z11 = (a + b*X11 + c*Y11 + d*X11*Y11)

    ax.plot_surface(X11, Y11, Z11,alpha =0.4)

    plt.show()
    # Plane plot ends


# Second order regression
# y = a + b x1 + c x2 + d x1x2 + e __ + f __

# To minimize the cost function and derive the linear constants (a,b,c,d,e,f)


if mess ==2:
    def gradient_descent(x1,x2,y):
        x1 = np.array(x1)
        x2 = np.array(x2)
        y = np.array(y)

        # Starting with some value of a,b,c,d and then take steps to minimize the
        # cost function
        a_curr = b_curr = c_curr = d_curr = e_curr = f_curr =  0
        iteartions = 1000
        learning_rate = 0.0000001
        n = len(x1)

        for i in range(iteartions):
            y_predicted = a_curr + b_curr*x1 + c_curr*x2 + d_curr *x1*x2 + e_curr*(x1**2) + f_curr*(x2**2)
            cost = (1/n) * sum([val**2 for val in (y-y_predicted)])

            ad = -(2 / n) * sum(y - y_predicted)
            bd = -(2/n)* sum((y-y_predicted)*x1)
            cd = -(2/n) * sum((y - y_predicted) * x2)
            dd = -(2/n) * sum((y - y_predicted) * x1*x2)
            ed = -(2/n) * sum((y - y_predicted) * (x1**2))
            fd = -(2 / n) * sum((y - y_predicted) * (x2** 2))


            a_curr = a_curr - learning_rate * ad
            b_curr = b_curr - learning_rate * bd
            c_curr = c_curr - learning_rate * cd
            d_curr = d_curr - learning_rate * dd
            e_curr = e_curr - learning_rate * ed
            f_curr = f_curr - learning_rate * fd



            #print("a {}, b {}, c {}, d {}, e {}, f {} , g{} ,cost {}, iteration {}".format(a_curr,b_curr,c_curr,d_curr,e_curr,f_curr,g_curr,cost,i))

        return [a_curr,b_curr,c_curr,d_curr,e_curr,f_curr]

    # Scatter plot 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X, Y, Z, c='r', marker='o')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    x1 = X
    x2 = Y
    y = Z
    # Scatter plot ends

    [a,b,c,d,e,f] = gradient_descent(x1,x2,y)

    # Plane plot 3D
    x11 = np.linspace(0,10,10)
    y11 = np.linspace(0,10,10)

    X11,Y11 = np.meshgrid(x11,y11)
    Z11 = (a + b*X11 + c*Y11 + d*X11*Y11 * e*X11*X11 + f*Y11*Y11 )

    ax.plot_surface(X11, Y11, Z11,alpha =0.4)

    plt.show()
    # Plane plot ends

# Third order regression
# y = a + b x1 + c x2 + d x1x2 + e __ + f __ + g __ + h ,i ,j

# To minimize the cost function and derive the linear constants (a,b,c,d,e,f,g,h,i,j)


if mess==3:
    def gradient_descent(x1,x2,y):
        x1 = np.array(x1)
        x2 = np.array(x2)
        y = np.array(y)

        # Starting with some value of a,b,c,d and then take steps to minimize the
        # cost function
        a_curr = b_curr = c_curr = d_curr = e_curr = f_curr = g_curr = h_curr = i_curr = j_curr = 0
        iteartions = 1000
        learning_rate = 0.000001
        n = len(x1)

        for x in range(iteartions):
            y_predicted = a_curr + b_curr*x1 + c_curr*x2 + d_curr *x1*x2 + e_curr*(x1**2) + f_curr*(x2**2) + g_curr*(x1)*(x2**2) + h_curr*(x1**2)*(x2) + i_curr*(x1**3) + + j_curr*(x2**3)
            cost = (1/n) * sum([val**2 for val in (y-y_predicted)])

            ad = -(2 / n) * sum(y - y_predicted)
            bd = -(2/n)* sum((y-y_predicted)*x1)
            cd = -(2/n) * sum((y - y_predicted) * x2)
            dd = -(2/n) * sum((y - y_predicted) * x1*x2)
            ed = -(2/n) * sum((y - y_predicted) * (x1**2))
            fd = -(2 / n) * sum((y - y_predicted) * (x2** 2))
            gd = -(2 / n) * sum((y - y_predicted) * (x1) * (x2**2))
            hd = -(2 / n) * sum((y - y_predicted) * (x2) * (x1 ** 2))
            id = -(2 / n) * sum((y - y_predicted) * (x1 ** 3))
            jd = -(2 / n) * sum((y - y_predicted) * (x2 ** 3))


            a_curr = a_curr - learning_rate * ad
            b_curr = b_curr - learning_rate * bd
            c_curr = c_curr - learning_rate * cd
            d_curr = d_curr - learning_rate * dd
            e_curr = e_curr - learning_rate * ed
            f_curr = f_curr - learning_rate * fd
            g_curr = g_curr - learning_rate * gd
            h_curr = h_curr - learning_rate * hd
            i_curr = i_curr - learning_rate * id
            j_curr = j_curr - learning_rate * jd


            #print("a {}, b {}, c {}, d {}, e {}, f {} , g{} ,cost {}, iteration {}".format(a_curr,b_curr,c_curr,d_curr,e_curr,f_curr,g_curr,cost,i))

        return [a_curr,b_curr,c_curr,d_curr,e_curr,f_curr,g_curr,h_curr,i_curr,j_curr]

    # Scatter plot 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X, Y, Z, c='r', marker='o')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    x1 = X
    x2 = Y
    y = Z
    # Scatter plot ends

    [a,b,c,d,e,f,g,h,i,j] = gradient_descent(x1,x2,y)

    # Plane plot 3D
    x11 = np.linspace(0,10,10)
    y11 = np.linspace(0,10,10)

    X11,Y11 = np.meshgrid(x11,y11)
    Z11 = (a + b*X11 + c*Y11 + d*X11*Y11 * e*X11*X11 + f*Y11*Y11 + g*Y11*Y11*X11 + h*X11*Y11*X11 + i*Y11*Y11*X11 + j*Y11*Y11*Y11)

    ax.plot_surface(X11, Y11, Z11,alpha =0.4)

    plt.show()
    # Plane plot ends


if mess==4:
    def gradient_descent(x1,x2,y):
        x1 = np.array(x1)
        x2 = np.array(x2)
        y = np.array(y)
    
        # Starting with some value of a,b,c,d and then take steps to minimize the
        # cost function
        a_curr = b_curr = c_curr = d_curr = e_curr = f_curr = g_curr = h_curr = i_curr = j_curr = k_curr = l_curr = m_curr = n_curr = o_curr = 0
        iteartions = 100
        learning_rate = 0.00000001
        n = len(x1)
    
        for x in range(iteartions):
            y_predicted = a_curr + b_curr*x1 + c_curr*x2 + d_curr *x1*x2 + e_curr*(x1**2) + f_curr*(x2**2) + g_curr*(x1)*(x2**2) + h_curr*(x1**2)*(x2) + i_curr*(x1**3) + + j_curr*(x2**3) \
                          + k_curr*(x1**2)*(x2**2) +  l_curr*(x1**3)*(x2**1) + +  m_curr*(x1**1)*(x2**3) +  n_curr*(x1**4) + o_curr*(x2**4)
    
            cost = (1/n) * sum([val**2 for val in (y-y_predicted)])
    
            ad = -(2 / n) * sum(y - y_predicted)
            bd = -(2/n)* sum((y-y_predicted)*x1)
            cd = -(2/n) * sum((y - y_predicted) * x2)
            dd = -(2/n) * sum((y - y_predicted) * x1*x2)
            ed = -(2/n) * sum((y - y_predicted) * (x1**2))
            fd = -(2 / n) * sum((y - y_predicted) * (x2** 2))
            gd = -(2 / n) * sum((y - y_predicted) * (x1) * (x2**2))
            hd = -(2 / n) * sum((y - y_predicted) * (x2) * (x1 ** 2))
            id = -(2 / n) * sum((y - y_predicted) * (x1 ** 3))
            jd = -(2 / n) * sum((y - y_predicted) * (x2 ** 3))
            kd = -(2 / n) * sum((y - y_predicted) * (x1 ** 2)*(x2**2))
            ld = -(2 / n) * sum((y - y_predicted) * (x1 ** 1) * (x2 ** 3))
            md = -(2 / n) * sum((y - y_predicted) * (x1 **3) * (x2 ** 1))
            nd = -(2 / n) * sum((y - y_predicted) * (x1 ** 4))
            od = -(2 / n) * sum((y - y_predicted) * (x2 ** 4))
    
    
            a_curr = a_curr - learning_rate * ad
            b_curr = b_curr - learning_rate * bd
            c_curr = c_curr - learning_rate * cd
            d_curr = d_curr - learning_rate * dd
            e_curr = e_curr - learning_rate * ed
            f_curr = f_curr - learning_rate * fd
            g_curr = g_curr - learning_rate * gd
            h_curr = h_curr - learning_rate * hd
            i_curr = i_curr - learning_rate * id
            j_curr = j_curr - learning_rate * jd
    
            k_curr = k_curr - learning_rate * kd
            l_curr = l_curr - learning_rate * ld
            m_curr = m_curr - learning_rate * md
            n_curr = n_curr - learning_rate * nd
            o_curr = o_curr - learning_rate * od
    
    
    
            #print("a {}, b {}, c {}, d {}, e {}, f {} , g{} ,cost {}, iteration {}".format(a_curr,b_curr,c_curr,d_curr,e_curr,f_curr,g_curr,cost,i))
    
        return [a_curr,b_curr,c_curr,d_curr,e_curr,f_curr,g_curr,h_curr,i_curr,j_curr,k_curr,l_curr,m_curr,n_curr,o_curr]
    
    # Scatter plot 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(X, Y, Z, c='r', marker='o')
    
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    
    x1 = X
    x2 = Y
    y = Z
    # Scatter plot ends
    
    [a,b,c,d,e,f,g,h,i,j,k,l,m,n,o] = gradient_descent(x1,x2,y)
    
    # Plane plot 3D
    x11 = np.linspace(0,10,10)
    y11 = np.linspace(0,10,10)
    
    X11,Y11 = np.meshgrid(x11,y11)
    Z11 = (a + b*X11 + c*Y11 + d*X11*Y11 * e*X11*X11 + f*Y11*Y11 + g*Y11*Y11*X11 + h*X11*Y11*X11 + i*Y11*Y11*X11 + j*Y11*Y11*Y11 + k*X11*X11*Y11*Y11 + l*X11*Y11*Y11*Y11 + m*X11*X11*X11*Y11
           + n*X11 * X11 * X11 * X11 + o*Y11*Y11*Y11*Y11)
    
    ax.plot_surface(X11, Y11, Z11,alpha =0.4)
    
    plt.show()
    # Plane plot ends

