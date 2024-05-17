from pyomo.environ import *
import numpy as np
# Define model
model = ConcreteModel()

# Sets
segments = list(range(1, 25))

# Parameters
lambda_values = {i: 40 + 2*i for i in segments}  # Example: Linear increase in price for simplicity
q_limits = {i: 10*i for i in segments}  # Example: Linearly increasing maximum quantity for simplicity
Q = 240  # Example: Maximum quantity

# Variables
model.q = Var(within=NonNegativeReals)
model.p = Var(within=NonNegativeReals)  # If needed in constraints or objective
model.y = Var(segments, within=Binary)
model.lambda_ = Var(within=NonNegativeReals)

# Objective function
def objective_rule(model):
    return model.lambda_ * model.q
model.obj = Objective(rule=objective_rule, sense=maximize)

# Constraints
def quantity_constraint_rule(model):
    return model.q <= Q
model.quantity_con = Constraint(rule=quantity_constraint_rule)

def binary_sum_rule(model):
    return sum(model.y[i] for i in segments) == 1
model.binary_sum_con = Constraint(rule=binary_sum_rule)

def lambda_rule(model):
    return model.lambda_ == sum(lambda_values[i] * model.y[i] for i in segments)
model.lambda_con = Constraint(rule=lambda_rule)

def q_segment_rule(model, i):
    return model.q <= q_limits[i] * model.y[i]
model.q_segment_con = Constraint(segments, rule=q_segment_rule)

# Solve the model
opt = SolverFactory('gurobi')
result = opt.solve(model)

# Display results
print("Optimal Bids:")
print(f"Optimal Quantity (q): {model.q.value}")
print(f"Optimal Price (p): {model.p.value}")  # If price is included
print("Optimal Binary Variables (y_i):")
for i in segments:
    print(f"Segment {i}: y_{i} = {model.y[i].value}")

optimal_price = sum(lambda_values[i] * model.y[i].value for i in segments)
optimal_quantity = model.q.value
optimal_profit = optimal_price * optimal_quantity

print(f"Optimal Clearing Price (lambda): {optimal_price}")
print(f"Optimal Quantity: {optimal_quantity}")
print(f"Optimal Profit: {optimal_profit}")
