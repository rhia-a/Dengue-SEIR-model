
import numpy as np
from  scipy.integrate import odeint
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pandas as pd

"""
SEIR model using 2006 Veracruz data, fitting of beta
"""

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
beta0=2e-6 #Baseline infection rate 
sigma=0.2 #Exposed to infection rate (~5 days)
gamma=0.1 #Recovery rate (~10 days)
chi= 0.00003753 #natural birth and death rate (1/73)/365)) avg life expectancy - 73.7

t = np.linspace(0,365,366) #time points /day over three years

def carrying_capacity (t, k0, epsilon, phi):
   """
   carrying_capacity (mosquito numbers varying with seasonality)
   - k0: baseline carrying capacity of mosquitoes
   - epsilon: seasonality amplitude
   - phi: what day of the year carrying capacity peaks
   """
   return k0 * (1 + epsilon * np.cos(2 * np.pi* (t-phi)/365)) 

def beta_t (beta0, M):
    """
    beta_t (infection rate at given time dependent on mosquitoes)
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

    Kt = carrying_capacity(t, k0, epsilon, phi)

    beta = beta_t(beta0, M)

    dMdt = r * M * (1-M/Kt) - mu * M
    dSdt = chi * N -beta * S * I / N -chi * S
    dEdt = beta * S * I / N - sigma * E - chi * E
    dIdt = sigma * E - gamma * I - chi * I
    dRdt = gamma * I - chi * R
    return dMdt, dSdt, dEdt, dIdt, dRdt #returning dydt

y_0= M_0, S_0, E_0, I_0, R_0 #set initial conditions

def model (beta0): 

    ret = odeint(seir_m_model, y_0, t, args=(N, r, k0, epsilon, phi, mu, beta0, sigma, gamma, chi)) #solves ode using parameters from initial conditions returning at all t values
    M, S, E, I, R = ret.T # T is transpose. Swaps rows and columns so 4 rows

    weekly_I = []

    for i in range (len(cases)):
        start = i * 7
        end = start + 7
        weekly_I.append(np.sum(I[start:end]))

    return np.array(weekly_I)

weekly_I = model(beta0)

def objective(scaled_beta):
    beta = scaled_beta[0] * 1e-6
    weekly_I = model(beta)
    rss = np.sum((cases - weekly_I)**2)
    return rss

#initial_guess = [2.0]       # represents beta = 2e-6
#bounds = [(0.01, 100)]      # represents beta in [1e-8, 1e-4]

for start in [0.5, 1.0, 2.0, 5.0, 10.0]:
    result = minimize(objective, [start], bounds=[(0.01, 100)], method='L-BFGS-B')
    print(f"start={start}: best beta={result.x[0]*1e-6:.3e}, RSS={result.fun:.1f}, nit={result.nit}")
#result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')

best_beta = result.x[0] * 1e-6
print(result.success, result.message, result.nit)
print("Best beta:", best_beta)
print("Minimum RSS:", result.fun)

"""
def objective (beta):
    weekly_I = model(beta[0])
    rss = np.sum((cases - weekly_I)**2) #calculating the error
    return rss

initial_guess = [2e-6]
bounds = [(1e-8, 1e-4)]
result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
print ("Best beta0:", result.x[0])
print("Minimum RSS:", result.fun)

best_beta = result.x[0]
weekly_I = model(best_beta)
"""

plt.figure(figsize=(8,5))
plt.plot(weeks, cases, 'ro', label='Observed')
plt.plot(weeks, weekly_I, 'b-', label='Model')
plt.legend()
plt.xlabel("Week")
plt.ylabel("Cases")
plt.title("Observed vs Model Veracruz Dengue Cases (2006)")
plt.show()


#plotting the SEIR against time
"""
axes[0].plot(t, S, 'b', label='Susceptible')
axes[0].plot(t, E, 'y', label='Exposed')
axes[0].plot(t, I, 'r', label='Infected')
axes[0].plot(t, R, 'g', label='Recovered')
axes[0].set_xlabel('Time (days)')
axes[0].set_ylabel('Population')
axes[0].set_title('SEIR with mosquitoes Model Simulation')
axes[0].legend()
axes[0].grid()

#plotting mosquitoes against time
axes[1].plot(t, M, color='purple', label= "Mosquitoes")
axes[1].set_xlabel('Time (days)')
axes[1].set_ylabel('Population')
axes[1].set_title('SEIR with mosquitoes Model Simulation')
axes[1].legend()
axes[1].grid()
plt.show()
"""