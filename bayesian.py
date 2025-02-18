from collections import defaultdict
from functools import reduce
from itertools import product

# Nick Pucci CS465 HW1 Part 1

class Bool(int):
    """
    Just displays True and False as T and F for better readability
    """
    __str__ = __repr__ = lambda self: 'T' if self else 'F'

T = Bool(True)
F = Bool(False)


CPTable = {
    #single probabilties 
    'A' : 0.87, #P(A=True)
    'B' : 0.62, #P(B=True)
    'E' : 0.95, #P(E=True)
    'F' : 0.29, #P(F=True)

    #conditional probabilities:
    'C' : { #P(C=True | A, B)
        (T, T): 0.18,
        (T, F): 0.06,
        (F, T): 0.98,
        (F, F): 0.35
    },
    'D' : { #P(D=True | C)
        T : 0.46,
        F : 0.03
    },
    'G' : { #P(G=True | D, E, F)
        (T, T, T): 0.32,
        (F, T, T): 0.01,
        (T, F, T): 0.48,
        (F, F, T): 0.07,
        (T, T, F): 0.21,
        (F, T, F): 0.45,
        (T, F, F): 0.76,
        (F, F, F): 0.19
    },
    'H' : { #P(H=True | G)
        T : 0.28,
        F : 0.79
    },
    'I' : { #P(I=True | C)
        T : 0.12,
        F : 0.34
    },
    'J' : { #P(J=True | C)
        T : 0.91,
        F : 0.56
    }
}

PARENTS = { #hardcode for creating parent relationships
    'A': [],
    'B': [],
    'E': [],
    'F': [],
    'C': ['A', 'B'],
    'D': ['C'],
    'G': ['D', 'E', 'F'],
    'H': ['G'],
    'I': ['C'],
    'J': ['C']
}

def get_user_query():
    """
    Prompt user for a query, convert into dictionary so we can use it
    Example input: "A=False, C=False, G=True"
    Returns: {'A': False, 'C': False, 'G': True}
    """

    user_input = input("query: ").strip()

    query = {}

    for assignment in user_input.split(','):
        var, val = assignment.strip().split('=')
        query[var] = T if val.strip().lower() == 'true' else F #converting string to a boolean value (returns F if 'false != true')
        
    return query

def get_missing_vars(query):
    """
    Create list of variables that were missing from the query
    Example input: {'A': False, 'C': False, 'G': True}
    Returns: ['B', 'D', 'E', 'F', 'H', 'I', 'J']
    """
    missing_vars = [] #start with empty list for missing variables from query
    for var in CPTable.keys(): #for each variable that is given in the query
        if var not in query: #check if this variable exists in the CPTable. If it doesn't, append to missing_vars list
            missing_vars.append(var)
    return missing_vars

def get_parents(X):
    """ 
    Returns the list of parents for a given variable from the Bayesian Network. 
    """
    return PARENTS.get(X, [])  # if X not found, returns []

def get_joint_probability(query):
    """
    Compute P(query) by summing over all possible values of missing variables.
    """
    missing_vars = get_missing_vars(query)

    # if no missing variables, return the joint probability directly
    if not missing_vars:
        return compute_joint(query)

    total_prob = 0.0

    # generate all possible assignments for missing variables by iterating through each assignment
    for assignment in product([T, F], repeat=len(missing_vars)):
        extended_query = query.copy()
        for var, value in zip(missing_vars, assignment):
            extended_query[var] = value  #assigning values to missing vars
        
        total_prob += compute_joint(extended_query) #add completed joint probability computation of the curr assignment

    return total_prob

def compute_joint(query):
    """
    Computes the joint probability of a fully specified assignment.
    """
    equation = []
    
    for var, value in query.items():
        if var in {'A', 'B', 'E', 'F'}:  # Independent variables
            prob = CPTable[var] if value == T else 1 - CPTable[var]
        else:  # Conditional probabilities
            parent_values = tuple(query[parent] for parent in get_parents(var))
            if len(parent_values) == 1:
                parent_values = parent_values[0]  # avoid tuple for single parent
            prob = CPTable[var][parent_values] if value == T else 1 - CPTable[var][parent_values]
        
        equation.append(prob)

    return reduce(lambda x, y: x * y, equation, 1.0)      




#Testing

# query = {'A': T, 'C': T, 'D': T, 'G': T}  # Example input
# print("P(A=True, C=False, G=True) =", get_joint_probability(query))


query = get_user_query()
print(get_joint_probability(query))

# print("Parsed query: ", query)
# print("Missing vars: ", get_missing_vars(query))
# print("Number of missing vars: ", len(get_missing_vars(query)))
# #var = 'B'
# #print("Parent(s) of given variable: ", get_parents(var))
# print("Joint probability: ", get_joint_probability(query))
# #ALL VARIABLES: A=True, B=False, C=True, D=False, E=False, F=False, G=True, H=True, I=False, J=True
# #MISSING VARIABLES: A=True, B=False, G=True
# #MISSING VARIABLES: A=True, C=False, G=True, H=False
# #MISSING VARIABLES: A=True, B=True, C=True, E=True, F=True, G=True, J=True
# #MISSING VARIABLES: A=True

