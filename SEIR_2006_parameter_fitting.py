
import numpy as np
from  scipy.integrate import odeint
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pandas as pd

"""
SEIR model using 2006 Veracruz data, fitting of beta
"""

#reads the excel data and extracts the 2 columns I need
data = pd.read_excel("Mexico data.xlsx", sheet_name = "Veracruz_2006") 
weeks = data["Week"] 
cases = data["dengue_total_cases"]

M_0 = 10_000 # initial number of mosquitoes 
S0 =1000 #initial number of susceptible
E0 = 5 #initial number of exposed 
I0 = 3 #initial number of infected 
R0 = 0 #initial number of recovered
N = S0+E0+I0+R0 #total population 
#7,110,214 - population from 2005 census

r=0.6 #mosquito growth rate
k0=10**5 #carrying capacity of mosquitoes
epsilon=0.6 #seasonality amplitude
phi=290 #what day of the year the carrying capacity peaks (oct 15)
mu=0.05 #mosquito death rate (~20 days)
beta0= 2e-6 #Baseline infection rate 
sigma=0.18 #Exposed to infection rate (~5.5 days)
gamma=0.1 #Recovery rate (~10 days)
chi= 0.00003753 #natural birth and death rate (1/73)/365)) avg life expectancy - 73.7

t = np.linspace(0,365,366) #time points /day over one year

def carrying_capacity (t, k0, epsilon, phi):
   """
   carrying_capacity (mosquito numbers varying with seasonality - calcs it for each day)
   - k0: baseline carrying capacity of mosquitoes
   - epsilon: seasonality amplitude
   - phi: what day of the year carrying capacity peaks
   """
   return k0 * (1 + epsilon * np.cos(2 * np.pi* (t-phi)/365)) 

def beta_t (beta0, M):
    """
    beta_t (infection rate at given time dependent on mosquitoes, more = higher force of infection)
    - beta0: baseline infection rate
    - M: number of mosquitoes
    """
    return beta0 * M 


def seir_m_model(y, t, N, r, k0, epsilon, phi, mu, beta0, sigma, gamma, chi):
    """
    SEIR_m model
    - r: mosquito growth rate
    - K0: baseline mosquito carrying capacity
    - epsilon: seasonality amplitude
    - phi: what day of the year carrying capacity peaks
    - mu: mosquito death rate
    - beta0: baseline infection rate
    - sigma: exposed to infected rate
    - gamma: recovery rate
    """
    M, S, E, I, R = y 

    Kt = carrying_capacity(t, k0, epsilon, phi) #find day n Kt

    beta = beta_t(beta0, M) #find day n beta

    dMdt = r * M * (1-M/Kt) - mu * M
    dSdt = chi * N -beta * S * I / N -chi * S
    dEdt = beta * S * I / N - sigma * E - chi * E
    dIdt = sigma * E - gamma * I - chi * I
    dRdt = gamma * I - chi * R
    return dMdt, dSdt, dEdt, dIdt, dRdt #returning dydt

y_0= M_0, S0, E0, I0, R0 #set initial conditions

def model (beta0, epsilon,phi,mu,r,sigma,gamma, E0, I0, R0): #solving the model for these parameters

    S0 = N - E0 - I0 - R0 # keeps total pop constant
    y_0 = (M_0, S0, E0, I0, R0) #initial values that go into odeint

    ret = odeint(seir_m_model, y_0, t, args=(N, r, k0, epsilon, phi, mu, beta0, sigma, gamma, chi)) #solves ode using parameters from initial conditions returning at all t values
    M, S, E, I, R = ret.T # T is transpose. Swaps rows and columns 

    weekly_I = [] #creates an empty list

    for i in range (len(cases)):  
        start = i * 7
        end = start + 7
        weekly_I.append(np.sum(I[start:end]))  #runs the model for each week

    return np.array(weekly_I)

def objective(params):

    scaled_beta, epsilon, phi, mu, r, sigma, gamma, E0, I0, R0 = params #defining my parameters
    beta = scaled_beta*1e-6 #scaling beta as scipy works better with bigger numbers
    weekly_I = model(beta,epsilon,phi,mu,r,sigma,gamma, E0, I0, R0)
    rss = np.sum((cases-weekly_I)**2)
    return rss

initial_guess = [
     2,           #beta
     0.6,         #epsilon
     290,         #phi
     0.05,        #mu
     0.6,         #r
     0.1,         #sigma
     0.2,         #gamma
     5,           #E0
     3,           #I0
     0            #R0
     ]

bounds = [
    (0.01,100),    #beta
    (0,1),         #epsilon
    (0,365),       #phi
    (0,1),         #mu
    (0,1),         #r
    (0,1),         #sigma
    (0,1),         #gamma
    (0,30),        #E0
    (0,30),        #I0
    (0,30)         #R0
    ]
result = minimize(objective,initial_guess,bounds=bounds,method="L-BFGS-B")
scaled_beta, best_epsilon, best_phi, best_mu, best_r, best_sigma, best_gamma, best_E0, best_I0, best_R0 = result.x
best_beta = scaled_beta*1e-6

print("Best beta =",best_beta)  #prints best values after optimisation for all parameters
print("Best epsilon =",best_epsilon)
print("Best phi =",best_phi)
print("Best mu =",best_mu)
print("Best r =",best_r)
print("Best sigma =",best_sigma)
print("Best gamma =",best_gamma)
print("Best E0 =",best_E0)
print("Best I0 =",best_I0)
print("Best R0 =",best_R0)

print("RSS =",result.fun) #RSS
rmse = np.sqrt(result.fun/len(cases))
print("RMSE =",rmse) #RMSE (how many cases off per week)
tss = np.sum((cases-np.mean(cases))**2)
r2 = 1-result.fun/tss
print("R² =",r2) #quantifies RSS, explains variation
print(result.x)
print(result.success)
print(result.message)
print(result.nit)

weekly_I = model(best_beta,best_epsilon,best_phi,best_mu,best_r,best_sigma,best_gamma,best_E0,best_I0,best_R0)

#plots observed vs model using fixed parameters
plt.figure(figsize=(8,5))
plt.plot(weeks, cases, 'ro', label='Observed')
plt.plot(weeks, weekly_I, 'b-', label='Model')
plt.legend()
plt.xlabel("Week")
plt.ylabel("Cases")
plt.title("Observed vs Model Veracruz Dengue Cases (2006)")
plt.show()

