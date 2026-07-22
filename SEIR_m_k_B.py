
import numpy as np
from  scipy.integrate import odeint
import matplotlib.pyplot as plt

"""
SEIR model with beta0 added in 
"""

M_0 = 10_000 # initial number of mosquitoes 
S_0 =1000 #initial number of susceptible
E_0 = 5#initial number of exposed 
I_0 = 2 #initial number of infected 
R_0 = 0 #initial number of recovered
N = S_0+E_0+I_0+R_0 #total population 

r=0.6 #mosquito growth rate
k0=10**5 #carrying capacity of mosquitoes
epsilon=0.6 #seasonality amplitude
phi=244 #what day of the year the carrying capacity peaks (sep 1)
mu=0.05 #mosquito death rate (~20 days)
beta0=2e-6 #Baseline infection rate 
sigma=0.2 #Exposed to infection rate (~5 days)
gamma=0.1 #Recovery rate (~10 days)
chi= 0.01 #natural birth and death rate 

t = np.linspace(0,1460,1461) #time points /day over three years

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


def seir_m_model(y, t, N, r, k0, epsilon, phi, mu, beta0, sigma, gamma):
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

ret = odeint(seir_m_model, y_0, t, args=(N, r, k0, epsilon, phi, mu, beta0, sigma, gamma)) #solves ode using parameters from initial conditions returning at all t values
print(ret.shape)
M, S, E, I, R = ret.T # T is transpose. Swaps rows and columns so 4 rows

#creating 2 subplots
fig,axes = plt.subplots(1,2, figsize=(20, 12))

#plotting the SEIR against time
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