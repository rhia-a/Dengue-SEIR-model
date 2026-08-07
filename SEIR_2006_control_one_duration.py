
import numpy as np
from  scipy.integrate import odeint
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pandas as pd

"""
SEIR model using 2006 Veracruz data, with larval control
Fixed fitted parameters
"""

#reads the excel data and extracts the 2 columns I need
data = pd.read_excel("Mexico data.xlsx", sheet_name = "Veracruz_2006") 
weeks = data["Week"] 
cases = data["dengue_total_cases"]

#Initial conditions
M_0 = 0.40 # relative initial abundance of mosquitoes 
E0 = 5 #initial number of exposed
I0 = 3 #initial number of infected
R0 = 0 #initial number of recovered

#population
N = 7_110_214 #population from 2005 census
S0 = N - E0 - I0 - R0 

#Fixed parameters
k0=1 #realtive carrying capacity of mosquitoes
mu=0.04 #mosquito death rate (avg lifespan of ~25 days) 
sigma=0.1818 #Exposed to infection rate (~5.5 days) 
gamma=0.1 #Recovery rate (~10 days) 
chi= 0.00003753 #natural birth and death rate (1/73)/365)) avg life expectancy - 73.7
phi = 200 # what day of year carrying capacity peaks
rho = 0.25 #reporting_rate 

#Fitted parameters from optimisiation
beta0 = 0.139181286266903 # Baseline infection rate 
epsilon = 0.8302327164594038 #seasonality amplitude
r = 0.9848641778053943 #mosquito growth rate

#control parameters
c = 0.4 #strength of larval control
start_day = 150
duration = 14

t = np.linspace(0,365,366) #time points /day over one year

def larval_control(t,c, start_day, duration):

    if start_day <= t <= start_day + duration:
        return c
    else:
        return 0


"""
   carrying_capacity (mosquito numbers varying with seasonality - calcs it for each day)
   - k0: baseline carrying capacity of mosquitoes
   - epsilon: seasonality amplitude
   - phi: what day of the year carrying capacity peaks
"""
def carrying_capacity (t, k0, epsilon, phi, c, start_day, duration):
    control = larval_control (t,c,start_day,duration)
    return (1 - control) * k0 * (1 + epsilon * np.cos(2 * np.pi* (t-phi)/365)) 
    

def beta_t (beta0, M):
    """
    beta_t (infection rate at given time dependent on mosquitoes, more = higher force of infection)
    - beta0: baseline infection rate
    - M: number of mosquitoes
    """
    return beta0 * M 


def seir_m_model(y, t, N, r, k0, epsilon, phi, mu, beta0, sigma, gamma, chi, c, start_day, duration):
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

    Kt = carrying_capacity(t, k0, epsilon, phi, c, start_day, duration) #find day n Kt

    beta = beta_t(beta0, M) #find day n beta

    dMdt = r * M * (1-M/Kt) - mu * M
    dSdt = chi * N -beta * S * I / N -chi * S
    dEdt = beta * S * I / N - sigma * E - chi * E
    dIdt = sigma * E - gamma * I - chi * I
    dRdt = gamma * I - chi * R
    return dMdt, dSdt, dEdt, dIdt, dRdt #returning dydt

y_0= M_0, S0, E0, I0, R0 #set initial conditions

def model (c, start_day, duration): #solving the model for these parameters

    S0 = N - E0 - I0 - R0 # keeps total pop constant
    y_0 = (M_0, S0, E0, I0, R0) #initial values that go into odeint

    ret = odeint(seir_m_model, y_0, t, args=(N, r, k0, epsilon, phi, mu, beta0, sigma, gamma, chi, c, start_day, duration)) #solves ode using parameters from initial conditions returning at all t values
    M, S, E, I, R = ret.T # T is transpose. Swaps rows and columns 

    daily_cases = rho * sigma * E #measuring sigmaE rather than I

    weekly_cases = [] #creates an empty list

    for i in range (len(cases)):  
        start = i * 7
        end = start + 7
        weekly_cases.append(np.sum(daily_cases[start:end]))  #runs the model for each week
    weekly_cases = np.array(weekly_cases)

    return weekly_cases, M

durations = [7, 14, 21, 28, 35, 42]


#test different intervention start days
start_days = range(1, 366)   # test every possible starting day
peak_list = []
total_list = []
duration_results = []

for duration in durations:

    peak_list = []
    total_list = []

    start_days = range(1, 366)

    for start_day in start_days:

        weekly_cases, M = model(c, start_day, duration)

        peak_cases = np.max(weekly_cases)
        total_cases = np.sum(weekly_cases)

        peak_list.append(peak_cases)
        total_list.append(total_cases)

    best_peak_index = np.argmin(peak_list)
    best_total_index = np.argmin(total_list)

    duration_results.append({
        "Duration": duration,
        "Best start day (peak)": list(start_days)[best_peak_index],
        "Minimum peak cases": peak_list[best_peak_index],
        "Best start day (total)": list(start_days)[best_total_index],
        "Minimum total cases": total_list[best_total_index]
    })
    
duration_results = pd.DataFrame(duration_results)
print(duration_results)

plt.figure(figsize=(8,5))

plt.plot(
    duration_results["Duration"],
    duration_results["Minimum peak cases"],
    marker="o"
)

plt.xlabel("Intervention duration (days)")
plt.ylabel("Minimum peak weekly cases")
plt.title("Effect of intervention duration on peak dengue cases")
plt.grid(True)

plt.show()

plt.figure(figsize=(8,5))

plt.plot(
    duration_results["Duration"],
    duration_results["Minimum total cases"],
    marker="o"
)

plt.xlabel("Intervention duration (days)")
plt.ylabel("Minimum total annual cases")
plt.title("Effect of intervention duration on total dengue cases")
plt.grid(True)

plt.show()