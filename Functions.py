import numpy as np
import numpy as np
import matplotlib.pyplot as plt



#Specified parameters in task

A = 1
F_s = 10**6
T = 1/F_s
n_0 = -256
N = 513
f_0 = 10**5
omega_0 = 2*np.pi*f_0
phi = np.pi/8 #Radians
phi_deg = (np.pi*180)/(np.pi*8) #Degrees

#Function for finding the mean square error
def meanSquareErr(a,b):
    error = 0
    n = len(a)
    for i in range(n):
        diff = (a[i]-b[i])
        squared_diff = diff**2
        error+=squared_diff
    mse = error/n
    return mse


#To find the frequency of interest in the signal, MLE in task a) argMax omega_hat = (2*pi*m^*)/TM
def argMax(fft,period,length):
    max_value = max(fft)
    index_of_max_value = np.where(fft == max_value)[0][0] #[0][0] in the end to slice out the index of interest from a numpy array
    omega_estimate = 2*np.pi*index_of_max_value * (1/(period*length)) #From the formula in task 

    return omega_estimate, index_of_max_value

#Function to generate the signal given in the task x[n] = s[n] + w[n]
def generate_signal(sigma_squared):   


    rwgn = np.random.normal(0,np.sqrt(sigma_squared), size = N)     #Generate white Gaussian noise, real and imaginary respectively
    iwgn = np.random.normal(0,np.sqrt(sigma_squared), size = N)*1j
    
    
    #Gaussian white noise
    gwn = []                                #The noise given in the task contains both real and imaginary noise, so these are added together in a list --> w[n] = gwn
    for i in range(N):
        gwn.append(rwgn[i]+iwgn[i])

        
    s_n =[]                         # Discrete exponential signal of interest s[n]
    for n in range(N):
        s_n.append(A*np.exp(complex(0,1)*((omega_0)*(n + n_0)*T + phi)))

        
    total_signal=[]                    #Total signal x[n] consisting of the signal of interest added with gaussian complex noise x[n] = s[n] + w[n]
    for i in range(N):
        total_signal.append(s_n[i] + gwn[i])

    return total_signal

