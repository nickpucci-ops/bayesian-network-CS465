import bnlearn as bn
import pandas as pd
import numpy as np
from pgmpy.factors.discrete import TabularCPD

# Define the Bayesian Network Structure as a list of edges
dag = [
    ('0', '4'),
    ('1', '2'),
    ('1', '5'),
    ('2', '4'),
    ('3', '4')
]

# Build Bayesian Network
model = bn.make_DAG(dag)

# Convert the bnlearn model to a pgmpy Bayesian Network model
pgmpy_model = model['model']

# Define CPDs in TabularCPD format
cpd_0 = TabularCPD(variable='0', variable_card=2, values=[[0.64], [0.36]])
cpd_1 = TabularCPD(variable='1', variable_card=2, values=[[0.6], [0.4]])
cpd_2 = TabularCPD(variable='2', variable_card=2, 
                   values=[[0.17, 0.3], [0.83, 0.7]], 
                   evidence=['1'], evidence_card=[2])
cpd_3 = TabularCPD(variable='3', variable_card=2, values=[[0.6], [0.4]])
cpd_4 = TabularCPD(variable='4', variable_card=2,
                   values=[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                           [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]],
                   evidence=['0', '2', '3'], evidence_card=[2, 2, 2])
cpd_5 = TabularCPD(variable='5', variable_card=2,
                   values=[[0.17, 0.3], [0.83, 0.7]], 
                   evidence=['1'], evidence_card=[2])

# Add CPDs to the pgmpy model
pgmpy_model.add_cpds(cpd_0, cpd_1, cpd_2, cpd_3, cpd_4, cpd_5)

# Validate the model
assert pgmpy_model.check_model()

# Plot the DAG
bn.plot(model)


print("-------------------Beginning Testing---------------------")
data_sizes = [1000000]
for n in data_sizes:
    print("----------------------Sampling------------------------")
    df = bn.sampling(model, n=n)
    
    # Learn structure from sampled data
    print("-----------------------Learn structure from sampled data------------------------")
    learned_structure = bn.structure_learning.fit(df)
    
    # Learn parameters using learned structure
    print("------------------------Learn parameters from structure------------------------")
    learned_model = bn.parameter_learning.fit(learned_structure, df)
    
    # Learn parameters using original structure
    print("----------------------Learn parameters using original structure------------------------")
    original_model_w_params = bn.parameter_learning.fit(model, df)
    
    print(f"Completed learning for n={n} samples\n")
