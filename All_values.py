

from time import sleep
import numpy as np
import matplotlib.pyplot as plt
import statistics as stats
import scipy.optimize as spo
#import Calculations as calc
import Functions as func
import pandas as pd

#EASY TO RUN script for estimating frequencies and phase using method a) and frequencies for method b) for all the SNR values (and FFT sizes in method a)
#File to generate a table of frequency estimates and phase estimates in method a) with all the different SNR values and different FFT-lengths
#Also generates table with different SNR for method b)  

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
fixed_length_nelder = 2**10 #Uses a fixed fft size of 2^10 to make finetunes from 

#Values for user to decide. NOTE: Takes extremely long time to run through many iterations. For instance like 20 minutes to run throug 1000, so recommended to run maximum 100, this takes long enough
iterations_mle = 5
iterations_nelder = 3 #Takes long time with many iterations so recommend picking a small number to get results fast


def ML_estimations(signal,fftsize): #Takes the signal x[n] and exponent in length parameters #Bør kalle Maximum_Likelihood elns
     

    DFT_signal = np.fft.fft(signal, fftsize) #Discrete fourier transform of the signal
    

    

    d_omega, index = func.argMax(np.abs(DFT_signal),T,fftsize) #Finding dominant frequency and its index
    d_freq = d_omega/(2*np.pi)
    ang = np.angle(np.exp(-(complex(0,1))*d_omega*n_0*T)*DFT_signal[index], deg=True)

    return d_freq, ang




def iterate_MLE(): #Iterates over all the SNRs and FFT sizes
    freq_means = []
    freq_error_variances = []
    ang_means = []
    ang_error_variances = []
    CRLB_freqs = []
    CRLB_angs = []
    length_str = []
    db = []
    


    for i in range(10,21,2):
        length = 2**i             #Lengths of 2**k der k = {10,12,14,16,18,20}
        str_length = "2^"+str(i)  #Lengths in 2^k as string for excel sheet
        for j in range(-10,61,10): #Decibels in the range {-10,0,10,10,20,40,60}
            SNR_db = j
            db.append(SNR_db)
            length_str.append(str_length)
            SNR_lin = 10**(SNR_db/10)
            sigma_squared = (A**2)/(2*SNR_lin)
            P = (N*(N-1))/2
            Q = (N*(N-1)*(2*N-1))/6
            CRLB_omega_0 = (12*sigma_squared)/((A**2)*(T**2)*N*((N**2)-1))
            CRLB_f_0 = (CRLB_omega_0)/(2*np.pi)**2
            CRLB_phi = (12*sigma_squared*((n_0**2)*N+2*n_0*P+Q))/((A**2)*(N**2)*((N**2)-1))
            CRLB_phi_deg = (CRLB_phi*180)/np.pi
            CRLB_freqs.append(CRLB_f_0)
            CRLB_angs.append(CRLB_phi_deg)

            freqs = []
            angs = []
            freq_errors = []
            ang_errors = []

            for i in range(iterations_mle):   #iteration estimates to find mean, variance of errors etc of each length and SNR
                x_n = func.generate_signal(sigma_squared)
                freq, ang = ML_estimations(x_n,length)
                freq_error = f_0-freq
                ang_error = phi_deg-ang
                freqs.append(freq)
                freq_errors.append(freq_error)
                angs.append(ang)
                ang_errors.append(ang_error)

            freq_mean = stats.mean(freqs)
            freq_means.append(freq_mean)
            freq_error_variance = stats.variance(freq_errors)
            freq_error_variances.append(freq_error_variance)
            ang_mean = stats.mean(angs)
            ang_means.append(ang_mean)
            ang_error_variance = stats.variance(ang_errors)  
            ang_error_variances.append(ang_error_variance)
            

    return freq_means, freq_error_variances, ang_means, ang_error_variances, CRLB_freqs, CRLB_angs, length_str, db


def iterate_nelder(): #Iterates over the different SNR values with fixed FFT size = 2^10
    
    nelder_means = []
    nelder_vars_errors = []
    CRLB_freqs = []
    dB = []

    for j in range(-10,61,10):
        dB.append(j)
        SNR_db = j
        SNR_lin = 10**(SNR_db/10)           
        sigma_squared = (A**2)/(2*SNR_lin)
        P = (N*(N-1))/2
        Q = (N*(N-1)*(2*N-1))/6
        CRLB_omega_0 = (12*sigma_squared)/((A**2)*(T**2)*N*((N**2)-1))
        CRLB_f_0 = (CRLB_omega_0)/(2*np.pi)**2
        CRLB_freqs.append(CRLB_f_0)

        def minimization(x):
            x_n = func.generate_signal(sigma_squared)
            oFFT = np.fft.fft(x_n,fixed_length_nelder)
            
            s = []
            for i in range(N):
                s.append(A*np.exp(complex(0,1)*((2*np.pi*x[0])*(i+n_0)*T+phi)))
                
            aFFT = np.fft.fft(s,fixed_length_nelder)
            min_sq_err = func.meanSquareErr(np.abs(aFFT),np.abs(oFFT))
            return min_sq_err


        nelder_freqs = []
        err_nelder_freq = []
        for i in range(iterations_nelder):

             result = spo.optimize._minimize_neldermead(minimization,100000) 
             nelder_freq = result.x[0]
             nelder_freqs.append(nelder_freq)
             error = f_0 - nelder_freq
             err_nelder_freq.append(error)
             
        
             
        mean_nelder_freq = stats.mean(nelder_freqs)
        nelder_means.append(mean_nelder_freq)
        var_err_nelder_freq = stats.variance(err_nelder_freq)
        nelder_vars_errors.append(var_err_nelder_freq)

        
        
    return nelder_means, nelder_vars_errors, dB        
       




def main():
    
    print(f"Calculating a dataframe showing the average frequency estimate from {iterations_mle} iterations")
    print("with SNRs of -10 throug 60, in steps of 10dB for FFT-sizes, M = 2^k, k ∈ {10,12,14,16,18,20}")
    f_means, f_err_variances, phi_means, phi_err_variances, f_CRLB, phi_CRLB, fft_length, decibels = iterate_MLE()
    d_freq = {"Length of FFT" : fft_length, "SNR [dB]" : decibels, "Frequency average [Hz]" : f_means, "Variance of frequency estimate error [Hz^2]" : f_err_variances, "CRLB [Hz^2]" : f_CRLB}
    d_f = pd.DataFrame(data=d_freq)
    print("Table of frequency estimates, and variances of frequency estimation errors below (This might take some time depending on number of iterations):")
    print()
    print(d_f)
    print()
    print(f"Calculating a dataframe showing the average phase estimate with the same amount of iterations and values for SNR and FFT-size")

    d_ang = {"Length of FFT" : fft_length, "SNR [dB]" : decibels, "Phase average [Degrees]" : phi_means, "Variance of phase estimate error [Degrees^2]" : phi_err_variances, "CRLB [Degrees^2]" : phi_CRLB}
    d_a = pd.DataFrame(data=d_ang)
    print("Table of phase estimates, and variance of phase estimation errors below: ")
    print()
    print(d_a)
    save = str(input("Do you want to save the tables into an excel-file?[y/n]"))
    if save == "y":
        file_name_freq = "Frequency_sheet_"+str(iterations_mle)+"_iterations.xls"
        file_name_phase = "Phase_sheet_"+str(iterations_mle)+"_iterations.xls"
        d_f.to_excel(file_name_freq)
        d_a.to_excel(file_name_phase)
        print(f"Frequency and phase estimates of {iterations_mle} iterations succesfully exported into Excel files {file_name_freq} and {file_name_phase}")
    else:
        print("Thats totally fine")

    sleep(2)
    print("\n")

    print(f"Doing part b with {iterations_nelder} iterations:")
    print(f"A dataframe showing the average frequency of {iterations_nelder} finetunes for the decibel levels of -10 through 60,\n in steps of 10dB shown below. This might take some time depending on number of iterations")
    print()
    average_nelder, var_err_nelder, dB = iterate_nelder()

    d_nelder = {"SNR [dB]": dB, "Frequency average [Hz]": average_nelder, "Variance of frequency estimate error [Hz^2]": var_err_nelder}
    d_n = pd.DataFrame(data=d_nelder)
    print(d_n)

    save_nelder = str(input("Do you want to save the tables into an excel-file?[y/n]"))
    if save_nelder == "y":
        filename_nelder = "Frequency_sheet_nelder_"+str(iterations_nelder)+"_iterations.xls"
        d_n.to_excel(filename_nelder)
        print(f"Frequency estimates of {iterations_nelder} iterations succesfully exported into Excel file {filename_nelder}")
    else:
        print("Thats okay too")


main()
    

                


    








