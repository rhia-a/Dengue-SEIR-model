
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

y_0= M_0, S0, E0, I0, R0 #set initial conditions

def model (beta0, epsilon,phi,mu,r,gamma,sigma, E0, I0, R0): #solving the model for these parameters

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

weekly_I = model(beta0,epsilon,phi,mu,r, gamma, sigma, E0, I0, R0) #runs model using my initial guess

def objective_beta(scaled_beta):  #optimising beta
    beta = scaled_beta[0] * 1e-6 #scaling it as larger numbers are easier for scipy
    weekly_I = model(beta,epsilon, phi,mu,r,gamma,sigma, E0, I0, R0) #runs model
    rss = np.sum((cases - weekly_I)**2) #calc error (least squares regression)
    return rss

best_result = None

for start in [0.5,1,2,5,10]: #testing different initial beta values
    result = minimize(objective_beta, [start], bounds=[(0.01, 100)], method='L-BFGS-B') #min, bounds, method
    print(f"start={start}: beta={result.x[0]*1e-6:.3e}, RSS={result.fun:.1f}, nit={result.nit}")

    if best_result is None or result.fun < best_result.fun:
        best_result = result #picks result with lowest RSS 

best_beta = best_result.x[0] * 1e-6 #converts scaled back into transmission rate
print("Best beta:", best_beta)
print("Minimum RSS:", best_result.fun)

beta0 = best_beta # fixes beta0

def objective_epsilon(epsilon): #repeat steps for epsilon now.

    weekly_I = model(beta0, epsilon[0], phi, mu, r, gamma, sigma, E0, I0, R0)
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

epsilon = best_epsilon # fixes epsilon

def objective_phi(phi): #repeat steps for phi

    weekly_I = model(beta0, epsilon, phi[0], mu, r, gamma, sigma, E0, I0, R0)
    rss = np.sum((cases - weekly_I)**2)
    return rss
    
best_result = None

for start in [50,100,150,200,250,300,350]: #testing different initial phi values
    result = minimize(objective_phi, [start], bounds=[(0, 365)], method='L-BFGS-B')
    print(f"start={start}: phi={result.x[0]}, RSS={result.fun:.1f}, nit={result.nit}")

    if best_result is None or result.fun < best_result.fun:
        best_result = result

best_phi = best_result.x[0]
print("Best phi:", best_phi)
print("Minimum RSS:", best_result.fun)

phi = best_phi # fixes phi

def objective_mu(mu): # repeats steps for mu

    weekly_I = model(beta0, epsilon, phi, mu[0], r, gamma, sigma, E0, I0, R0)
    rss = np.sum((cases - weekly_I)**2)
    return rss
    
best_result = None

for start in [0.02, 0.04, 0.06, 0.08]: #testing different initial mu values
    result = minimize(objective_mu, [start], bounds=[(0, 10)], method='L-BFGS-B')
    print(f"start={start}: mu={result.x[0]}, RSS={result.fun:.1f}, nit={result.nit}")

    if best_result is None or result.fun < best_result.fun:
        best_result = result

best_mu = best_result.x[0]
print("Best mu:", best_mu)
print("Minimum RSS:", best_result.fun)

mu = best_mu # fixes mu

def objective_r(r): # repeat steps for r

    weekly_I = model(beta0, epsilon, phi, mu, r[0], gamma, sigma, E0, I0, R0)
    rss = np.sum((cases - weekly_I)**2)
    return rss
    
best_result = None

for start in [0.2, 0.4, 0.6, 0.8]: #testing different initial r values
    result = minimize(objective_r, [start], bounds=[(0, 1)], method='L-BFGS-B')
    print(f"start={start}: rho={result.x[0]}, RSS={result.fun:.1f}, nit={result.nit}")

    if best_result is None or result.fun < best_result.fun:
        best_result = result

best_r = best_result.x[0]
print("Best rho:", best_r)
print("Minimum RSS:", best_result.fun)

r = best_r # fixes r

def objective_gamma(gamma): # repeat steps for gamma

    weekly_I = model(beta0, epsilon, phi, mu, r, gamma[0], sigma, E0, I0, R0)
    rss = np.sum((cases - weekly_I)**2)
    return rss
    
best_result = None

for start in [0.05, 0.1, 0.15, 0.2]: #testing different initial gamma values
    result = minimize(objective_gamma, [start], bounds=[(0, 1)], method='L-BFGS-B')
    print(f"start={start}: gamma={result.x[0]}, RSS={result.fun:.1f}, nit={result.nit}")

    if best_result is None or result.fun < best_result.fun:
        best_result = result

best_gamma = best_result.x[0]
print("Best gamma:", best_gamma)
print("Minimum RSS:", best_result.fun)

gamma = best_gamma # fixes gamma

def objective_sigma(sigma): # repeats steps for sigma

    weekly_I = model(beta0, epsilon, phi, mu, r, gamma, sigma[0], E0, I0, R0)
    rss = np.sum((cases - weekly_I)**2)
    return rss
    
best_result = None

for start in [0.1, 0.2, 0.4, 0.6, 0.8]: #testing different initial sigma values
    result = minimize(objective_sigma, [start], bounds=[(0, 1)], method='L-BFGS-B')
    print(f"start={start}: sigma={result.x[0]}, RSS={result.fun:.1f}, nit={result.nit}")

    if best_result is None or result.fun < best_result.fun:
        best_result = result

best_sigma = best_result.x[0]
print("Best sigma:", best_sigma)
print("Minimum RSS:", best_result.fun)

sigma = best_sigma #fixes sigma

def objective_initial(params): # repeat steps for E0,I0,R0

    E0 = params[0]
    I0 = params[1]
    R0 = params[2]

    weekly_I = model(beta0, epsilon, phi, mu, r, gamma, sigma, E0, I0, R0)
    rss = np.sum((cases - weekly_I)**2)
    return rss

initial_guess = [5,3,0]
bounds = [(0,30),(0,30),(0,30)]
result = minimize(objective_initial, initial_guess, bounds=bounds, method='L-BFGS-B')

best_result = result

best_E0 = best_result.x[0]
best_I0 = best_result.x[1]
best_R0 = best_result.x[2]

print("Best E0 =", best_E0)
print("Best I0 =", best_I0)
print("Best R0 =", best_R0)
print("RSS =", result.fun)


weekly_I = model(best_beta, best_epsilon, best_phi, best_mu, best_r, best_gamma, best_sigma, best_E0, best_I0, best_R0) #model with fixed parameters

"""
Kt = carrying_capacity(t, k0, epsilon, phi)

plt.figure(figsize=(8,4))
plt.plot(t, Kt)
plt.xlabel("Day")
plt.ylabel("Carrying capacity")
plt.title("Seasonal mosquito carrying capacity")
plt.show()
"""

#plots observed vs model using fixed parameters
plt.figure(figsize=(8,5))
plt.plot(weeks, cases, 'ro', label='Observed')
plt.plot(weeks, weekly_I, 'b-', label='Model')
plt.legend()
plt.xlabel("Week")
plt.ylabel("Cases")
plt.title("Observed vs Model Veracruz Dengue Cases (2006)")
plt.show()

