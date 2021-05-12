# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 09:44:50 2020

@author: talebf
"""

import numpy as np
from scipy.fftpack import fft
import matplotlib.pyplot as plt
import os
import os.path
from matplotlib import cm
from scipy import signal 
from scipy import interpolate
from scipy.optimize import curve_fit
#from scipy import asarray as ar,exp

## to plot the TD and FD signal
############################
os.chdir("CeramicSamples/ExtraSamples/Al2O3_2")
path=os.getcwd()
dirs=os.listdir(path)
c=300  ##um/ps
d=3091 #um thickness of sample
x=0

t=900

for i in dirs:
    if "ref" in i:
        
      input_data_r=i
      with open(input_data_r, 'r') as f:
            data_r=np.genfromtxt(f,comments="!", dtype="float", usecols=(0,1) )
            
      time_r=data_r[:,0]
      # time_r1[0]=time_r1[1]
      # time_r=np.zeros(len(time_r1), dtype=float)
      signal_r=np.zeros(len(time_r), dtype=float)
      r=0
      # for r in range(len(time_r1)):
      #     time_r[-r]=time_r1[r]
      time_r[0]=time_r[1]
      if max(time_r)<1:
          time_r=time_r*1E12
          
    
      signal_r=data_r[:,1] 
      # signal_r1[0]=signal_r1[1]
      rr=0
      
      # for rr in range(len(signal_r1)):
          
      #     signal_r[-rr]=signal_r1[rr]
      signal_r[0]=signal_r[1]
      DC_offsetr=np.mean(signal_r[0:100])
      signal_r =signal_r - DC_offsetr
      window=signal.boxcar(len(time_r))
      signal_r=window*signal_r
      signal_r[t:-1]=0
      
      if x==0:
          idx_r=int(len(dirs)/2)
          signal_r_array=np.zeros([len(time_r),idx_r], dtype=float)
          Amps_r_array=np.zeros([len(time_r),idx_r], dtype=float)
          Freq_r=np.zeros([len(time_r),idx_r], dtype=float)
          Phase_r_array=np.zeros([len(time_r),idx_r], dtype=float)
    

      amp_r=np.fft.fft(signal_r)
      freq_r=np.fft.fftfreq(len(signal_r), np.mean(np.diff(time_r)))
      phase_r=np.angle(amp_r)
      phase_r=np.unwrap(phase_r)
      signal_r_array[:, x]=signal_r[:]
      Amps_r_array[:, x]=abs(amp_r[:])
      Freq_r[:, x]=freq_r[:]
      Phase_r_array[:, x]=phase_r[:]
      # plt.plot(data_r[:,0],data_r[:,1], linewidth=3)
#      print("1")
      x=x+1
      
    else:
         continue


#######################

x1=0  
for  i1 in dirs:     
        
    if "0deg" in i1:
        input_data_s=i1
        
        with open(input_data_s, 'r') as f:
            data_s=np.genfromtxt(f,comments="!", dtype="float", usecols=(0,1) )
            
            
        time_s=data_s[:,0]
        
        # time_s=np.zeros(len(time_r1), dtype=float)
        # r=0
        # for r in range(len(time_s1)):
            
        #     time_s[-r]=time_s1[r]
        time_s[0]=time_s[1]
        if max(time_s)<1:
            time_s=time_s*1E12
            
        signal1=data_s[:,1]
        
        rr1=0
        # signal1=np.zeros(len(signal11), dtype=float)
        # for rr1 in range(len(signal11)):
          
        #     signal1[-rr1]=signal11[rr1] 
        signal1[0]=signal1[1]  
        DC_offset=np.mean(signal1[0:50])
        signal1 =signal1 - DC_offset
        windows=signal.boxcar(len(time_s))
        signal_s=windows*signal1
        signal_s[t:-1]=0

        
        if x1==0:
            idx=int(len(dirs)/2)
            signal_array=np.zeros([len(time_s),idx], dtype=float)
            Amps_array=np.zeros([ len(time_s),idx], dtype=float)
            Freqs=np.zeros([len(time_s),idx], dtype=float)
            Phase_s_array=np.zeros([len(time_s),idx], dtype=float)
        
        
        amp_s=np.fft.fft(signal_s)
        freq_s=np.fft.fftfreq(len(signal_s), np.mean(np.diff(time_s)))
        phase_s=np.angle(amp_s)
        phase_s=np.unwrap(phase_s)
        signal_array[:, x1]=signal_s[:]
        Amps_array[ :,x1]=abs(amp_s[:])
        Freqs[ :,x1]=freq_s[:]
        Phase_s_array[ :,x1]=phase_s[:]
        # plt.plot(data_s[:,0],data_s[:,1], linewidth=3)        

        x1=x1 +1 
        
    else:
        continue
 
ref_TD=np.mean(signal_r_array,axis=1)
ref_FD=np.mean(Amps_r_array,axis=1)
ref_Ph=np.mean(Phase_r_array,axis=1) 
      
sig_TD=np.mean(signal_array,axis=1)
sig_FD=np.mean(Amps_array,axis=1) 
sig_Ph=np.mean(Phase_s_array,axis=1) 

indx= (freq_s>=0) & (freq_s<2) 

plt.figure()  
plt.plot(time_r,ref_TD,time_s,sig_TD, linewidth=3)
plt.xlabel("Time[ps] ") 
plt.ylabel("Amplitude a.u.") 
plt.legend(["Reference", "Sample"])

plt.figure()
plt.plot(freq_s[indx],20*np.log10(ref_FD[indx]), freq_s[indx],20*np.log10(sig_FD[indx]), linewidth=2)
plt.xlabel("Frequency [THz] ") 
plt.ylabel("Intensity dB") 
plt.legend(["Reference", "Sample"])


Phase_diff1=ref_Ph-sig_Ph

plt.figure()
plt.plot(ref_Ph[indx])
plt.plot(sig_Ph[indx])
plt.plot(Phase_diff1[indx] )
plt.xlabel("Frequency [THz] ")
plt.ylabel("Phase-radian") 
plt.legend(["Reference", "Sample", 'difference'])
plt.show()

#############################################

#MaxR_val=np.max(ref_TD)
#MaxS_val=np.max(sig_TD)
#
#MaxR_idx=list(ref_TD).index(MaxR_val)
#MaxS_idx=list(sig_TD).index(MaxS_val)
#t1= time_r[MaxR_idx]
#t2=time_s[MaxS_idx]
#################
### caluclate n in Time Domain
#
#delt_t=t2-t1  ###ps
#c=300  ##um/ps
#d=7585 #um
#
#n_td=(delt_t*c/d)+1
#print("t=",delt_t, " n=", n_td)


#################
## frequency domain Calculation

idx=int(len(dirs)/2)
Phase_diff=np.zeros([ len(time_s),idx], dtype=float)
Amp_TF=np.zeros([ len(time_s),idx], dtype=float)
n_TF=np.zeros([ len(time_s),idx], dtype=float)
ph_fitN=np.zeros([ len(time_s)], dtype=float)
A_coeff_a=np.zeros([ len(time_s),idx], dtype=float)
K_Coeff_a=np.zeros([ len(time_s),idx], dtype=float)
PhaseN=np.zeros([ len(time_s),idx], dtype=float)

#### linner fit
def liner (x,m,b):
    return m*x+b


for k in range(idx):
    Phase_diff[:,k]=ref_Ph-Phase_s_array[:,k]## phase difference
    Amp_TF1=Amps_array[:,k]/ref_FD ### ampl
    Amp_TF[:,k]=Amp_TF1
    
    diff_ph=Phase_diff[:,k]
    indx2=(freq_s>= 0.05) & (freq_s<= 0.25) ####  fit range
#    xx1=list(freq_s).index(0.12)
#    xx2=list(freq_s).index(0.55)
    xx=30
    yy=128
    f=freq_s[xx:yy]
    y=diff_ph[xx:yy]

    popt,pcov=curve_fit(liner,f, y)
    opfit=liner(x, popt[0],popt[1])
    x=freq_s
    ph_fit=liner(x,*popt)
    ph_fitN[0:xx]=ph_fit[0:xx]
    ph_fitN[xx:yy]=diff_ph[xx:yy]
    ph_fitN[yy:-1]=ph_fit[yy:-1]
    ph_fit1=-popt[1]+ph_fitN[:]
    PhaseN[:,k]=ph_fit1
#    plt.figure()
#    plt.plot(freq_s[indx],ph_fit1[indx],freq_s[indx],diff_ph[indx] )
    
    n_FD= 1+(ph_fit1*c)/(2*np.pi*freq_s*d)
    n_TF[:,k]=n_FD #### refractive Index for each measurment
#    plt.figure()
#    plt.plot(freq_s[indx2],n_FD[indx2])
    
    K_Coeff=-(c/(d*4*np.pi*freq_s))*np.log((Amp_TF1*(1+n_FD)**2)/(4*n_FD))
    K_Coeff_a[:,k]=K_Coeff
    A_coeff=10000*(4*np.pi*freq_s*K_Coeff)/c #### 1/cm  c um/ps 
#    plt.figure()
#    plt.plot(freq_s[indx2],A_coeff[indx2])
    A_coeff_a[:,k]=A_coeff

PhaseN_mean=np.mean(PhaseN,axis=1)    
n_mean=np.mean(n_TF,axis=1)    
K_mean=np.mean(K_Coeff_a,axis=1)
A_mean=np.mean(A_coeff_a,axis=1)
n_STD=np.std(n_TF,axis=1)
A_STD=np.std(A_coeff_a,axis=1)

n_complex=n_mean+1j*K_mean
    
plt.figure()
plt.plot(freq_s[indx],Phase_diff1[indx],freq_s[indx],PhaseN_mean[indx])
plt.xlabel("Frequency [THz] ") 
plt.ylabel("Phase-radian") 
plt.legend([ 'difference', 'extrapolated']) 

indx3=(freq_s>= 0.2 ) & (freq_s<= 1.2)


font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 20,
        }
fig, ((ax0, ax2),(ax1, ax3))=plt.subplots(2,2)
##1 Time Domain Signal
ax0.plot(time_r[:t],ref_TD[:t],time_s[:t],sig_TD[:t],linewidth=3)
ax0.set_xlabel("Time[ps] ",fontdict=font) 
ax0.set_ylabel("Amplitude [a.u.]",fontdict=font) 
#ax0.set_xlim([150,350])
ax0.tick_params(axis='both', labelsize=16,width=3)
ax0.legend(["Reference", "Sample"])

## Frequency Domain FFT
ax1.plot(freq_s[indx],20*np.log10(ref_FD[indx]), freq_s[indx],20*np.log10(sig_FD[indx]), linewidth=3 )
ax1.set_xlabel("Frequency [THz] ",fontdict=font)
ax1.set_ylabel("Intensity [dB]",fontdict=font) 
ax1.tick_params(axis='both', labelsize=16,width=3)
ax1.legend(["Reference", "Sample"])

### Refractive Index and Extinction coefficient
yerr=n_STD[indx3]*2
ax2.plot(freq_s[indx3],n_mean[indx3], linewidth=3)##,,ecolor="red"
ax2.fill_between(freq_s[indx3],n_mean[indx3]-yerr,n_mean[indx3]+yerr,color='red', alpha=0.4 )
ax2.set_ylim([1.1,2.3])
ax2.set_xlabel("Frequency [THz] ",fontdict=font) 
ax2.set_ylabel("Refractive index",fontdict=font) 
ax2.tick_params(axis='both', labelsize=16,width=3)

ax21=ax2.twinx()
color='green'
ax21.set_ylabel("Extinction coefficient",color=color,fontdict=font)
ax21.plot(freq_s[indx3],K_mean[indx3],color=color, linewidth=3)
ax21.set_ylim([-0.01,1])
ax21.tick_params(axis='y', labelsize=16, width=3, labelcolor=color)

### Absorption Coeffiecient
yerr2=A_STD[indx3]
ax3.plot(freq_s[indx3],A_mean[indx3],marker='o', linewidth=3 )
ax3.fill_between(freq_s[indx3],A_mean[indx3]-yerr2,A_mean[indx3]+yerr2,color='red', alpha=0.4 )
#ax3.set_ylim([0.1,13])
ax3.set_xlabel("Frequency [THz] ",fontdict=font)
ax3.set_ylabel("Absorption Coeffiecient [1/cm]",fontdict=font) 
ax3.tick_params(axis='both', labelsize=16,width=3)

print("thickness=",d/1000,"mm")
print("n at 0.3 THz =" ,np.round(n_mean[60],3), "+/-", np.round(n_STD[60],4) )
print("n at 0.12 THz =" ,np.round(n_mean[24],3), "+/-", np.round(n_STD[24],4) )

print("absorption at 0.3 THz =" ,np.round(A_mean[60],3), "+/-",np.round(A_STD[60],3),"cm-1" )
print("absorption at 0.12 THz =" ,np.round(A_mean[24],3), "+/-", np.round(A_STD[24],3),"cm-1" )
#A_Coeff1=(np.abs(np.imag(n_sam1))*freq*1E12*4*np.pi)/c
plt.show()
#
#









         