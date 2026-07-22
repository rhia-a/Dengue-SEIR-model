
import numpy as np
from  scipy.integrate import odeint
import matplotlib.pyplot as plt

M_0 = 10_000 # initial number of mosquitoes 
S_0 =10000 #initial number of susceptible
E_0 = 500 #initial number of exposed 
I_0=250 #initial number of infected 
R_0=0 #initial number of recovered
N=S_0+E_0+I_0+R_0 #total population

r= 0.6 #mosquito growth rate
k= 10**5 #carrying capacity of mosquitoes
mu=0.05 #mosquito death rate
beta=0.4 #Infection rate
sigma=0.2 #Exposed to infection rate
gamma=0.1 #Recovery rate

t = np.linspace(0,100,100) #time points over 100 days

def seir_m_model(y, t, N, r, k, mu, beta, sigma, gamma):
    """
    SEIR_m model
    - r: mosquito growth rate
    - k: mosquito carrying capacity
    - mu: mosquito death rate
    - beta: infection rate
    - sigma: exposed to infected rate
    - gamma: recovery rate
    """
    M, S, E, I, R = y 
    dMdt = r * M * (1-M/k) - mu * M
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return dMdt, dSdt, dEdt, dIdt, dRdt #returning dydt

y_0= M_0, S_0, E_0, I_0, R_0 #set initial conditions

ret = odeint(seir_m_model, y_0, t, args=(N, r, k, mu, beta, sigma, gamma)) #solves ode using parameters from initial conditions returning at all t values
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