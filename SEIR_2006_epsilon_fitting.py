
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
S_0 =1000 #initial number of susceptible
E_0 = 5#initial number of exposed 
I_0 = 2 #initial number of infected 
R_0 = 0 #initial number of recovered
N = S_0+E_0+I_0+R_0 #total population 
#7,110,214 - population from 2005 census

r=0.6 #mosquito growth rate
k0=10**5 #carrying capacity of mosquitoes
epsilon=0.6 #seasonality amplitude
phi=290 #what day of the year the carrying capacity peaks (oct 15)
mu=0.05 #mosquito death rate (~20 days)
beta0= 2e-6 #Baseline infection rate 
sigma=0.2 #Exposed to infection rate (~5 days)
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

y_0= M_0, S_0, E_0, I_0, R_0 #set initial conditions

def model (beta0, epsilon): 

    ret = odeint(seir_m_model, y_0, t, args=(N, r, k0, epsilon, phi, mu, beta0, sigma, gamma, chi)) #solves ode using parameters from initial conditions returning at all t values
    M, S, E, I, R = ret.T # T is transpose. Swaps rows and columns so 4 rows

    weekly_I = [] #creates an empty list

    for i in range (len(cases)):  
        start = i * 7
        end = start + 7
        weekly_I.append(np.sum(I[start:end]))  #runs the model for each week

    return np.array(weekly_I)

weekly_I = model(beta0,epsilon) #runs model using my initial guess

def objective_beta(scaled_beta):  #optimising beta
    beta = scaled_beta[0] * 1e-6
    weekly_I = model(beta,epsilon)
    rss = np.sum((cases - weekly_I)**2) #calc error (least squares regression)
    return rss

best_result = None

for start in [0.5,1,2,5,10]: #testing different initial beta values
    result = minimize(objective_beta, [start], bounds=[(0.01, 100)], method='L-BFGS-B')
    print(f"start={start}: beta={result.x[0]*1e-6:.3e}, RSS={result.fun:.1f}, nit={result.nit}")

    if best_result is None or result.fun < best_result.fun:
        best_result = result

best_beta = best_result.x[0] * 1e-6 #converts scaled back into transmission rate
print("Best beta:", best_beta)
print("Minimum RSS:", best_result.fun)

beta0 = best_beta

def objective_epsilon(epsilon):

    weekly_I = model(beta0, epsilon[0])
    rss = np.sum((cases - weekly_I)**2)
    return rss
    
best_result = None

for start in [0.2,0.4,0.6,0.8]: #testing different initial epsilon values
    result = minimize(objective_epsilon, [start], bounds=[(0, 1)], method='L-BFGS-B')
    print(f"start={start}: epsilon={result.x[0]}, RSS={result.fun:.1f}, nit={result.nit}")

    if best_result is None or result.fun < best_result.fun:
        best_result = result

best_epsilon = best_result.x[0]
print("Best epsilon:", best_epsilon)
print("Minimum RSS:", best_result.fun)

weekly_I = model(best_beta, best_epsilon)

plt.figure(figsize=(8,5))
plt.plot(weeks, cases, 'ro', label='Observed')
plt.plot(weeks, weekly_I, 'b-', label='Model')
plt.legend()
plt.xlabel("Week")
plt.ylabel("Cases")
plt.title("Observed vs Model Veracruz Dengue Cases (2006)")
plt.show()

