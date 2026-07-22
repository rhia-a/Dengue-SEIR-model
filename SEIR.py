
import numpy as np
from  scipy.integrate import odeint
import matplotlib.pyplot as plt

S_0 =10000 #initial number of susceptible
E_0 = 1000 #initial number of exposed 
I_0=250 #initial number of infected 
R_0=0 #initial number of recovered
N=S_0+E_0+I_0+R_0 #total population


beta=0.4 #Infection rate
sigma=0.2 #Exposed to infection rate
gamma=0.1 #Recovery rate

t = np.linspace(0,100,100) #time points over 100 days

def seir_model(y, t, N, beta, sigma, gamma):
    """
    SEIR model

    - beta: infection rate
    - sigma: exposed to infected rate
    - gamma: recovery rate
    """
    S, E, I, R = y 
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return dSdt, dEdt, dIdt, dRdt #returning dydt

y_0= S_0, E_0, I_0, R_0 #set initial conditions

ret = odeint(seir_model, y_0, t, args=(N, beta, sigma, gamma)) #solves ode using parameters from initial conditions returning at all t values
print(ret.shape)
S, E, I, R = ret.T # T is transpose. Swaps rows and columns so 4 rows

#plotting the SEIR against time
plt.figure(figsize=(10, 6)) 
plt.plot(t, S, 'b', label='Susceptible')
plt.plot(t, E, 'y', label='Exposed')
plt.plot(t, I, 'r', label='Infected')
plt.plot(t, R, 'g', label='Recovered')
plt.xlabel('Time (days)')
plt.ylabel('Population')
plt.title('SEIR Model Simulation')
plt.legend()
plt.grid()
plt.show()