from quantum_particle import QuantumParticle
from qpso_curve_fit import QPSOCurveFit
import matplotlib.pyplot as plt
import sys
sys.path.insert(0,'../rebalancer')
from rebalancer import load_data


data,data2 = load_data(['SPY'],'2017-12-27','2003-01-01')

t = data[1:-1]
x = data[0:-2]

qpso = QPSOCurveFit(200,200)
result = qpso.run(x,t)

test = result.evaluate(x)

plt.plot(data)
plt.plot(test)
plt.show()

