import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
plt.rcParams.update({'font.size': 8})

dates = pd.date_range(start='1-1-2020', end='12-31-2020')

# Load data
Demand = pd.read_csv('ENTSOE_AT_2020/Demand/AT_2020.csv')
Generation = pd.read_csv('ENTSOE_AT_2020/Generation/AT_2020.csv')
InstalledCapacity = pd.read_csv('ENTSOE_AT_2020/InstalledCapacity/AT_2020.csv')
Prices = pd.read_csv('ENTSOE_AT_2020/Prices/AT_2020.csv')


data = pd.merge(Demand, Generation, on='Time', how='inner')
data = pd.merge(data, Prices, on='Time', how='inner')
print(data.head())

# Keep only the last two columns of InstalledCapacity
# InstalledCapacity = InstalledCapacity.iloc[:, -2:]
# print(InstalledCapacity.transpose())

# All Sources
# data['Generation'] = data[['Solar','WindOnShore','WindOffShore','Hydro','HydroStorage','HydroPumpedStorage','Marine','Nuclear','Geothermal','Biomass','Waste','OtherRenewable','Lignite','Coal','Gas','CoalGas','Oil','ShaleOil','Peat','Other']].sum(axis=1)
# Renewable Sources
data['Generation'] = data[['Solar','WindOnShore','WindOffShore','Hydro','HydroStorage','HydroPumpedStorage','Marine','OtherRenewable']].sum(axis=1)


X = data[['Demand', 'Generation']]  # Predictor variables
y = data['Price']  # Response variable

X = sm.add_constant(X)

# Fit the regression model
model = sm.OLS(y, X).fit()

# Print the summary of the regression
print(model.summary())

data['Time'] = pd.to_datetime(data['Time'])

# Plotting the observed vs predicted prices
plt.figure(figsize=(10, 5))
predicted_prices = model.predict(X)
plt.plot(data['Time'], y, label='Actual Prices')
plt.plot(data['Time'], predicted_prices,'--', label='Predicted Prices')
plt.xlabel('Time')
plt.ylabel('Electricity Price (EUR/MWh)')
plt.title('Actual vs Predicted Electricity Prices')

# Set xticks as dates
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())

plt.legend()
plt.show()