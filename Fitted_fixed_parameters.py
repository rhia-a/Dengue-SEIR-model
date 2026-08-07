
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

M_0 = 0.40 # relative initial abundance of mosquitoes 
S0 =7110214 #initial number of susceptible
E0 = 0 #initial number of exposed 
I0 = 0 #initial number of infected 
R0 = 0 #initial number of recovered
N = S0+E0+I0+R0 #total population
S0 = N - E0 - I0 - R0 
#7,110,214 - population from 2005 census

#Fixed parameters
k0=1 #realtive carrying capacity of mosquitoes
mu=0.04 #mosquito death rate (avg lifespan of ~25 days) 
sigma=0.1818 #Exposed to infection rate (~5.5 days) 
gamma=0.1 #Recovery rate (~10 days) 
chi= 0.00003753 #natural birth and death rate (1/73)/365)) avg life expectancy - 73.7
phi = 200 # what day of year carrying capacity peaks
rho = 0.25 #reporting_rate 
N = 7_110_214
M_0 = 0.41
R0 = 0 #initial number of recovered

#r = mosquito growth rate
#epsilon = seasonality amplitude
#beta0 = Baseline infection rate 


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

def model (beta0, epsilon,r, E0, I0): #solving the model for these parameters

    S0 = N - E0 - I0 - R0 # keeps total pop constant
    y_0 = (M_0, S0, E0, I0, R0) #initial values that go into odeint

    ret = odeint(seir_m_model, y_0, t, args=(N, r, k0, epsilon, phi, mu, beta0, sigma, gamma, chi)) #solves ode using parameters from initial conditions returning at all t values
    M, S, E, I, R = ret.T # T is transpose. Swaps rows and columns 
    
    weekly_cases = [] #creates an empty list

    daily_cases = rho * sigma * E #measuring sigmaE rather than I

    for i in range (len(cases)):  
        start = i * 7
        end = start + 7
        weekly_cases.append(np.sum(daily_cases[start:end]))  #runs the model for each week
    weekly_cases = np.array(weekly_cases)

    return np.array(weekly_cases)

def objective(params):

    beta0, epsilon, r, E0, I0 = params #defining my parameters
    model_cases = model(beta0,epsilon,r, E0, I0)
    rss = np.sum((cases-model_cases)**2) #calculating rss
    return rss

initial_guess = [
     1e-3,        #beta0
     0.6,         #epsilon
     0.6,         #r
     5,           #E0
     3,           #I0
     ]

bounds = [
    (1e-4,1),     #beta0
    (0,1),        #epsilon
    (0,1),        #r
    (0,100),      #E0
    (0,100),      #I0
    ]

result = minimize(objective,initial_guess,bounds=bounds,method="L-BFGS-B") #finding the best fit to the data for each parameter
best_beta0, best_epsilon, best_r, best_E0, best_I0= result.x

print("Best beta0 =",best_beta0)  #prints best values after optimisation for all parameters
print("Best epsilon =",best_epsilon)
print("Best phi =",phi)
print("Best r =",best_r)
print("Gamma =",gamma)
print("Best E0 =",best_E0)
print("Best I0 =",best_I0)
print("Mu =",mu)
print("Sigma =", sigma)
print("rho =", rho)


print("RSS =",result.fun) #RSS
rmse = np.sqrt(result.fun/len(cases))
print("RMSE =",rmse) #RMSE (how many cases off per week)
tss = np.sum((cases-np.mean(cases))**2)
r2 = 1-result.fun/tss
print("R² =",r2) #quantifies RSS, explains variation
print(result.x)
print(result.success)
print(result.message)
print(result.nit) #number of iterations

model_cases = model(best_beta0,best_epsilon,best_r,best_E0,best_I0)

S_initial = N - best_E0 - best_I0 - R0
y_best = (M_0,S_initial,best_E0,best_I0,R0)
ret = odeint(seir_m_model,y_best,t,args=(N,best_r,k0,best_epsilon, phi,mu,best_beta0,sigma,gamma,chi))
M, S, E, I, R = ret.T

#creating 2 subplots
fig,axes = plt.subplots(1,2, figsize=(20, 12))

#plots observed vs model using fixed parameters
axes[0].plot(weeks, cases, 'ro', label='Observed')
axes[0].plot(weeks, model_cases, 'b-', label='Model')
axes[0].legend()
axes[0].set_xlabel("Week")
axes[0].set_ylabel("Cases")
axes[0].set_title("Observed vs Model Veracruz Dengue Cases (2006)")

#plots daily mosquito abundance
axes[1].plot(t, M, 'g')
axes[1].set_xlabel("Day")
axes[1].set_ylabel("Mosquito abundance")
axes[1].set_title ("Daily Mosquito abundance")

plt.tight_layout()
plt.show()


