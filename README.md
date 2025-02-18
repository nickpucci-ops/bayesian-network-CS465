# bayesian-network-CS465
Part 1 Calculating joint probabilities from Bayesian Networks
Part 2 Parameter learning (reference at https://libraries.io/pypi/bnlearn and https://pypi.org/project/pgmpy)  
Goal: Investigate how well the structure can be learned from data.
```
Part 1 Bayesian Network CPT
A True=0.87
B True=0.62
C | A:True,B:True = 0.18
C | A:False,B:True = 0.98
C | A:True,B:False = 0.06
C | A:False,B:False = 0.35
D | C:True = 0.46
D | C:False = 0.03
E True=0.95
F True=0.29
G | D:True,E:True,F:True = 0.32
G | D:False,E:True,F:True = 0.01
G | D:True,E:False,F:True = 0.48
G | D:False,E:False,F:True = 0.07
G | D:True,E:True,F:False = 0.21
G | D:False,E:True,F:False = 0.45
G | D:True,E:False,F:False = 0.76
G | D:False,E:False,F:False = 0.19
H | G:True = 0.28
H | G:False = 0.79
I | C:True = 0.12
I | C:False = 0.34
J | C:True = 0.91
J | C:False = 0.56
 ```

Example query: (use all lower case)
►	A=False, B=True, C=True, D=True, E=False
►	D=True
