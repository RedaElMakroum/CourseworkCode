import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 15})
plt.rcParams["font.family"] = "Times New Roman"

# Load data from Excel file
df = pd.read_excel('Exercise1Data.xlsx')

# Filter data for summer and winter
df_summer = df.iloc[5832:8040]
df_winter = df.iloc[1465:3648]

# Define the dependent variable and the independent variables for summer
Y_summer = df_summer['Demand']
X_summer = df_summer[['Temp']]
X_summer = sm.add_constant(X_summer)

# Define the dependent variable and the independent variables for winter
Y_winter = df_winter['Demand']
X_winter = df_winter[['Temp']]
X_winter = sm.add_constant(X_winter)


# Model1 for summer
model1_summer = sm.OLS(Y_summer, X_summer)
results1_summer = model1_summer.fit()

# Model1 for winter
model1_winter = sm.OLS(Y_winter, X_winter)
results1_winter = model1_winter.fit()


# print(results1_Summer.summary())
# print(results1_Winter.summary())

def generate_ticks_and_labels(data):
    labels = [date if hour == 12 else None for date, hour in zip(data['Time'].dt.strftime('%b %d'), data['Time'].dt.hour)]
    ticks = range(len(labels))
    return ticks, labels

WinterWeek = df.iloc[2209:2376] # 1 week of data in winter from 1 Jan to 7 Jan
SummerWeek = df.iloc[7321:7488] # 1 week of data in summer from 1 Aug to 7 Aug

# Predictions for Winter Week
X1_WinterWeek = WinterWeek[['Temp']]
X1_WinterWeek = sm.add_constant(X1_WinterWeek)
Y_WinterWeek = WinterWeek['Demand']
predictions1_WinterWeek = results1_winter.predict(X1_WinterWeek)

# Predictions for Summer Week
X1_SummerWeek = SummerWeek[['Temp']]
X1_SummerWeek = sm.add_constant(X1_SummerWeek)
Y_SummerWeek = SummerWeek['Demand']
predictions1_SummerWeek = results1_summer.predict(X1_SummerWeek)

print(WinterWeek)
# Ensure that 'Time' column is of datetime type
WinterWeek['Time'] = pd.to_datetime(WinterWeek['Time'])

# Convert 'Time' column to "Mon Day" format and store it
WinterWeekDays = WinterWeek['Time'].dt.strftime('%b %d')
WinterWeekDays = WinterWeekDays.loc[WinterWeek['Time'].dt.hour == 0]

W_ticks, W_labels = generate_ticks_and_labels(WinterWeek)
S_ticks, S_labels = generate_ticks_and_labels(SummerWeek)

#Model 2

# Define the dependent variable and the independent variables for summer
Y2_summer = df_summer['Demand']

X2_summer = df_summer[['Temp', 'Hour','Hour2','Hour3']]
X2_summer = sm.add_constant(X2_summer)

# Define the dependent variable and the independent variables for winter
Y2_winter = df_winter['Demand']

X2_winter = df_winter[['Temp', 'Hour','Hour2','Hour3']]
X2_winter = sm.add_constant(X2_winter)

# Model2 for summer
model2_summer = sm.OLS(Y2_summer, X2_summer)
results2_summer = model2_summer.fit()

# Model2 for winter
model2_winter = sm.OLS(Y2_winter, X2_winter)
results2_winter = model2_winter.fit()

# print(results2_Summer.summary())
# print(results2_Winter.summary())


# Model 2
# Predictions for Winter Week
X2_WinterWeek = WinterWeek[['Temp','Hour','Hour2','Hour3']]
X2_WinterWeek = sm.add_constant(X2_WinterWeek)
Y_WinterWeek = WinterWeek['Demand']
predictions2_WinterWeek = results2_winter.predict(X2_WinterWeek)

# Predictions for Summer Week
X2_SummerWeek = SummerWeek[['Temp','Hour','Hour2','Hour3']]
X2_SummerWeek = sm.add_constant(X2_SummerWeek)
Y_SummerWeek = SummerWeek['Demand']
predictions2_SummerWeek = results2_summer.predict(X2_SummerWeek)


# Plot for Model1 summer
# plt.figure(figsize=(16, 6))
# plt.subplot(2, 1, 1)
# plt.scatter(range(len(Y_summer)), Y_summer, label='Data')
# plt.plot(range(len(Y_summer)), results1_summer.fittedvalues, color='red', label='OLS')
# plt.xlabel('Temp')
# plt.ylabel('Demand')
# plt.title('Model 1 Summer: Demand vs Temp')
# plt.legend()

# # Plot for Model1 winter
# plt.subplot(2, 1, 2)
# plt.scatter(df_winter['Temp'], Y_winter, label='Data')
# plt.plot(df_winter['Temp'], results1_winter.fittedvalues, color='blue', label='OLS')
# plt.xlabel('Temp')
# plt.ylabel('Demand')
# plt.title('Model 1 Winter: Demand vs Temp')
# plt.legend()

# Plot for Model1 winter week
plt.close('all')
plt.figure(figsize=(16, 6))
plt.subplot(2, 1, 1)
plt.plot(range(len(Y_WinterWeek)), Y_WinterWeek, label='Actual',color='grey')
plt.plot(range(len(Y_WinterWeek)), predictions1_WinterWeek, color='blue', label='Predicted')
W_ticks_reduced = [tick for tick, label in zip(W_ticks, W_labels) if label is not None]
W_labels_reduced = [label for label in W_labels if label is not None]
plt.tick_params(axis='x', length=0)
plt.xticks(W_ticks_reduced, W_labels_reduced)
plt.ylabel('Demand (MW)')
plt.title('Model 4-1: Winter Week')
plt.grid(True, alpha=0.5)
plt.legend()

# Plot for Model1 summer week
plt.subplot(2, 1, 2)
plt.plot(range(len(Y_SummerWeek)), Y_SummerWeek, label='Actual', color='grey')
plt.plot(range(len(Y_SummerWeek)), predictions1_SummerWeek, color='blue', label='Predicted')
S_ticks_reduced = [tick for tick, label in zip(S_ticks, S_labels) if label is not None]
S_labels_reduced = [label for label in S_labels if label is not None]
plt.xticks(S_ticks_reduced, S_labels_reduced)
plt.tick_params(axis='x', length=0)
plt.ylabel('Demand (MW)')
plt.title('Model 4-1: Summer Week')
plt.grid(True, alpha=0.5)
plt.legend()

plt.tight_layout
plt.savefig("Figures\Task11Model4-1.svg", format='svg', bbox_inches='tight')


# Plot for Model2 for winter: winter week
plt.close('all')
plt.figure(figsize=(16, 6))
plt.subplot(2, 1, 1)
plt.plot(range(len(Y_WinterWeek)), Y_WinterWeek, label='Actual', color='grey')
plt.plot(range(len(Y_WinterWeek)), predictions2_WinterWeek, color='blue', label='Predicted')
plt.xticks(W_ticks_reduced, W_labels_reduced)
plt.tick_params(axis='x', length=0)
plt.ylabel('Demand (MW)')
plt.title('Model 4-2: Winter Week')
plt.grid(True, alpha=0.5)
plt.legend()

# Plot for Model2 for summer: summer week
plt.subplot(2, 1, 2)
plt.plot(range(len(Y_SummerWeek)), Y_SummerWeek, label='Actual', color='grey')
plt.plot(range(len(Y_SummerWeek)), predictions2_SummerWeek, color='blue', label='Predicted')
plt.xticks(S_ticks_reduced, S_labels_reduced)
plt.tick_params(axis='x', length=0)
plt.ylabel('Demand (MW)')
plt.title('Model 4-2: Summer Week')
plt.grid(True, alpha=0.5)
plt.legend()

plt.tight_layout()
plt.savefig("Figures\Task11Model4-2.svg", format='svg', bbox_inches='tight')


# # Plot for Model2 summer
# plt.figure(figsize=(16, 6))
# plt.subplot(1, 2, 1)
# plt.scatter(Y2_summer, results2_summer.fittedvalues)
# plt.plot([Y2_summer.min(), Y2_summer.max()], [Y2_summer.min(), Y2_summer.max()], color='red')
# plt.xlabel('Actual')
# plt.ylabel('Predicted')
# plt.title('Model 2 Summer: Actual vs Predicted')
# plt.legend()

# # Plot for Model2 winter
# plt.subplot(1, 2, 2)
# plt.scatter(Y2_winter, results2_winter.fittedvalues)
# plt.plot([Y2_winter.min(), Y2_winter.max()], [Y2_winter.min(), Y2_winter.max()], color='blue')
# plt.xlabel('Actual')
# plt.ylabel('Predicted')
# plt.title('Model 2 Winter: Actual vs Predicted')
# plt.legend()
# plt.show()