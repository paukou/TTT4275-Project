import statistics as stats
import Functions as func
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as spo


#EASY TO RUN SCRIPT WITH fixed values for SNR and fft-size

#Specified parameters from task

A = 1
F_s = 10**6
T = 1/F_s
n_0 = -256
N = 513
f_0 = 10**5
omega_0 = 2*np.pi*f_0
phi = np.pi/8 #Radians
phi_deg = (np.pi*180)/(np.pi*8) #Degrees

#----Values for the user to change if desired-------

SNR_db = -10  #Change this value to get a different SNR
FFT_size = 2**10 #----------------||-----------------FFT size
SNR_lin = 10**(SNR_db/10)
sigma_squared = (A**2)/(2*SNR_lin)
iterations = 100



#CRLB parameters

P = (N*(N-1))/2
Q = (N*(N-1)*(2*N-1))/6

#CRLB for the respective parameters omega_o and phi

CRLB_omega_0 = (12*sigma_squared)/((A**2)*(T**2)*N*((N**2)-1))
CRLB_phi = (12*sigma_squared*((n_0**2)*N+2*n_0*P+Q))/((A**2)*(N**2)*((N**2)-1))


#Operates with frequency so make a variable for the cramer lower bound for this parameter. 
#Formula for variance yields Hz^2 
CRLB_f_0 = (CRLB_omega_0)/(2*np.pi)**2

#CRLB for degrees
CRLB_phi_deg = (CRLB_phi*180)/np.pi

#Function to generate the signal given in the task x[n] = s[n] + w[n]
#Method a
def generate_signal():


    rwgn = np.random.normal(0,np.sqrt(sigma_squared), size = N)     #Generate white Gaussian noise, real and imaginary respectively
    iwgn = np.random.normal(0,np.sqrt(sigma_squared), size = N)*1j
    
    
    #Gaussian white noise
    gwn = []                                #The noise given in the task contains both real and imaginary noise, so these are added together in a list --> w[n] = gwn
    for i in range(N):
        gwn.append(rwgn[i]+iwgn[i])

        
    s_n =[]                         # Discrete exponential signal of interest s[n]
    for i in range(N):
        s_n.append(A*np.exp(complex(0,1)*((omega_0)*(i + n_0)*T + phi)))

        
    total_signal=[]                    #Total signal x[n] consisting of the signal of interest added with gaussian complex noise x[n] = s[n] + w[n]
    for i in range(N):
        total_signal.append(s_n[i] + gwn[i])

    return total_signal    #

#To find the frequency of interest in the signal, MLE in task a) argMax (over omega_hat) = (2*pi*m^*)/TM

def argMax(fft,period,length):
    max_value = max(fft)
    index_of_max_value = np.where(fft == max_value)[0][0] #[0][0] in the end to slice out the index of interest from a numpy array
    omega_estimate = 2*np.pi*index_of_max_value * (1/(period*length)) #From the formula in task 

    return omega_estimate, index_of_max_value


def estimations():
    x_n = generate_signal() #Total signal

    DFT_x = np.fft.fft(x_n, FFT_size) #Discrete fourier transform
    

    d_omega, index = argMax(np.abs(DFT_x),T,FFT_size) #Finding dominant frequency and its index
    d_freq = d_omega/(2*np.pi)                      #Converting from omega to frequency
    ang = np.angle(np.exp(-(complex(0,1))*d_omega*n_0*T)*DFT_x[index], deg=True) #From the formula in the task

    return d_freq, ang


#Method b
signal_with_noise = generate_signal()
fixed_length = 2**10
FFT_with_noise = np.fft.fft(signal_with_noise,fixed_length)
frequency_FFT_with_noise = argMax(np.abs(FFT_with_noise),T,fixed_length)[0]/(2*np.pi)

def minimization(x):
    signal_without_noise = []
    for i in range(N): 
        signal_without_noise.append(A*np.exp(complex(0,1)*((2*np.pi*x[0])*(i+n_0)*T+phi))) #Creating an exponential signal without noise
    
    FFT_without_noise = np.fft.fft(signal_without_noise,fixed_length)

    mean_square_error = func.meanSquareErr(np.abs(FFT_without_noise),np.abs(FFT_with_noise))
    return mean_square_error




def main():

    print(f"Firstly we investigate the performance of the FFT-based MLE (method a), doing {iterations} iterations with")
    print(f"SNR: {SNR_db} dB and FFT size: {FFT_size}")

    listfreqs = []
    listangles = []
    freq_errors = []
    phi_errors = []
    
    for i in range(iterations):
        f, p = estimations()
        freq_error = f_0-f
        freq_errors.append(freq_error)
        phi_error = phi_deg-p 
        phi_errors.append(phi_error)
        listfreqs.append(f)
        listangles.append(p)
    
    average_freq = stats.mean(listfreqs)
    average_phase = stats.mean(listangles)
    var_err_freq = stats.variance(freq_errors)
    var_err_phase = stats.variance(phi_errors)
        
    
    print(f"Original frequency of signal: {f_0} Hz")
    print(f"Original phase of signal: {phi_deg} degrees")
    print()
    print(f"Cramer rao lower bound of frequency estimator: {CRLB_f_0} Hz^2")
    print(f"Cramer rao lower bound of phase estimator: {CRLB_phi_deg} degrees^2")
    print("\n")

    print(f"Here are the results:")
    print("\n")
    print(f"Average of estimated frequency: {average_freq} Hz")
    print(f"Variance of the frequency estimation error: {var_err_freq} Hz^2")
    print()
    print(f"Average of estimated phase: {average_phase} degrees")
    print(f"Variance of the phase estimation error: {var_err_phase} degrees^2")
    
    print("\n")
    print(f"Now evaluating the performance of the frequency estimate with method b)")
    print()

    

      
    result = spo.optimize._minimize_neldermead(minimization,100000)
    finetuned_frequency = result.x[0]
    
    print(f"The estimate of frequency with noise and FFT-size 2^10: {frequency_FFT_with_noise}")
    print()
    print(f"The estimate of frequency after finetuning: {finetuned_frequency}")
    print()
    diff_finetuned = np.abs(f_0-finetuned_frequency)
    diff_regular = np.abs(f_0-frequency_FFT_with_noise)
    if diff_finetuned < diff_regular:
        print(f"We observe that the deviation from the original frequency, {f_0}Hz, is less\nfor the finetuned estimate with an error of {diff_finetuned}Hz\ncompared to the 2^10 estimate without finetuning which has an error of {diff_regular}Hz")
    


main()
    











