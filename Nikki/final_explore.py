# imports:
import numpy as np
import pandas as pd

# visualizations:
import seaborn as sns
import matplotlib.pyplot as plt








#--------------Statistical Tests-------------#
# continous vs continous: Data is not normal: 
def eval_Spearmanresult(r,p,α=0.05):
    """
    
    """
    if p < α:
        return print(f"""We reject H₀, there appears to be a monotonic relationship.
Spearman's rs: {r:2f}.
P-value: {p}""")
    else:
        return print(f"""We fail to reject H₀: that there does not appear to be a monotonic relationship.
Spearman’s r: {r:2f}
P-value: {p}""")