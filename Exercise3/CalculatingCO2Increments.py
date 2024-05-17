from pyomo.environ import *  # noqa: F403
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


T = 24
timesteps = np.arange(T)

c_CO2 = 0 # EUR/tCO2
emissions_variable = np.zeros(20)

for i in range(1, 21):
    c_CO2 = - 10 + i*10
    # Note: does not correspond to all values in the description!
    thermalPlant = ['Coal', 'CCGT', 'Gas Turbine']
    power = {'Coal': 600,
            'CCGT': 400,
            'Gas Turbine': 300} # MW
    efficiency = {'Coal': 0.41,
            'CCGT': 0.58,
            'Gas Turbine': 0.4} 
    fuel_price = {'Coal': 10,
            'CCGT': 30,
            'Gas Turbine': 30} # EUR/MWhprim
    emission_factor = {'Coal': 0.35,
            'CCGT': 0.2,
            'Gas Turbine': 0.2} # tCO2/MWhprim
    MC = {} # marginal costs in EUR/MWh
    emissions = {} # emissions in tCO2/MWh
    for n in thermalPlant:
        MC[n] = (fuel_price[n] + emission_factor[n] * c_CO2) / efficiency[n]
        emissions[n] = emission_factor[n] / efficiency[n]
        
    # Load data
    df = pd.read_excel('Last_PV_Wind.xlsx')
    load = df['Last Winter [MW]'] # enter group number Summer/Winter here
    Wind = df['Wind 300 MW']
    PV = df['PV 200 MW Winter']
    
    # Create Pyomo model
    model = ConcreteModel()

    model.x = Var(thermalPlant, timesteps, within = NonNegativeReals)
    model.dual = Suffix(direction=Suffix.IMPORT)  # Enable dual information
    
    # Specify objective function!

    model.obj = Objective(
        expr = sum(model.x[n,t] * MC[n] for n in thermalPlant for t in timesteps) + 
            sum(model.x[n,t] * emissions[n] * c_CO2 for n in thermalPlant for t in timesteps),
        sense = minimize
    )

    def power_constraint_rule(model, n, t):    
     return model.x[n,t] <= power[n]
    def load_constraint_rule(model, t):    
     return sum(model.x[n,t] for n in thermalPlant) == load.loc[t] - Wind[t] - PV[t]


    model.power_con = Constraint(thermalPlant, timesteps, rule = power_constraint_rule)
    model.load_con = Constraint(timesteps, rule = load_constraint_rule)

    opt = SolverFactory('gurobi')
    opt_success = opt.solve(model)



    # model.display()

    # Get values of optimization variables
    PowerThermal = pd.DataFrame(index = timesteps, columns = thermalPlant)
    for t in timesteps:
        for n in thermalPlant:
            PowerThermal.loc[t, n] = model.x[n,t].value
        PowerThermal.loc[t, 'Wind'] = Wind[t]
        PowerThermal.loc[t, 'PV'] = PV[t]



    # Calculate total generation and shadow prices for each timestep
    total_generation = PowerThermal.sum(axis=1)
    shadow_prices = {t: model.dual[model.load_con[t]] for t in timesteps}


    # # Display the results
    # print("Shadow Prices for each hour (Optimal prices under competition):")
    # for t in timesteps:
    #     print(f"Hour {t}: {shadow_prices[t]:.2f} €/MWh")

    # # Plot the results
    # plt.figure(figsize=(12, 6))
    # plt.plot(timesteps, [shadow_prices[t] for t in timesteps], marker='o', linestyle='-')
    # plt.title('Optimal Electricity Price Over Time')
    # plt.xlabel('Time (hours)')
    # plt.ylabel('Price (€/MWh)')
    # plt.grid(True)
    # plt.show()

    # Plot

    # sns.set_theme(style='whitegrid')

    # fig, ax = plt.subplots()
    # ax.stackplot(timesteps, 
    #             PowerThermal.to_numpy(dtype = float).transpose(), 
    #             labels=thermalPlant + ['Wind', 'PV'])
    # ax.set_title('Power Plant Dispatch')
    # ax.legend(loc='upper left')
    # ax.set_ylabel('Generation [MW]')
    # ax.set_xlabel('Time [h]')
    # ax.set_xlim(xmin=timesteps[0], xmax=timesteps[-1])
    # fig.tight_layout()

    optimized_cost = model.obj()
    resulting_emissions = sum(model.x[n,t].value * emissions[n] for n in thermalPlant for t in timesteps)
    emissions_variable[i-1] = resulting_emissions
    # print("CO2 Prices: ", c_CO2)
    # print("Optimized Cost of Electricity: ", round(optimized_cost))
    # print("Resulting GHG Emissions: ", round(resulting_emissions))


print(emissions_variable)
plt.plot(np.arange(1, 21)*10, emissions_variable)
plt.title('Resulting GHG Emissions with increments of 10')
plt.xlabel('CO2 Price [EUR/tCO2]')
plt.ylabel('Emissions [tCO2]')
plt.grid(True, alpha = 0.5)
plt.show()