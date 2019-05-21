# a = 17938.0
# b = 8899.0
# c = 15162.0
# d = 35840.0
# x = float(sum([a, b, c, d]))

# print (round(100.0*float(a/x), 2))
# print (round(100.0*float(b/x), 2))
# print (round(100.0*float(c/x), 2))
# print (round(100.0*float(d/x), 2)) 

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D

# FEATURES

# indexes = ['GBc', 'RFc', 'MLPc', 'ABc', 'LRc', 'MLPmc', 'NBmc', 'DTmc']

# a = [0.7124, 0.6720, 0.6456, 0.6375, 0.6199, 0.7085, 0.6815, 0.6909]
# b = [0.7827, 0.7440, 0.7178, 0.6938, 0.6754, 0.7744, 0.7406, 0.7587]
# c = [0.371, 0.308, 0.262, 0.287, 0.255, 0.373, 0.365, 0.373]

# l1, = plt.plot(indexes, a, color='blue', label='Accuracy')
# l2, = plt.plot(indexes, b, color='green', label='F1')
# l3, = plt.plot(indexes, c, color='red', label='KS')

# plt.legend(handler_map={l1: HandlerLine2D(numpoints=2)})

# plt.ylabel('Metric Value')
# plt.xlabel('Classifiers')

# plt.axis([0, 7, 0.2, 0.8])
# plt.show()

indexes = ['GBc', 'RFc', 'MLPc', 'ABc', 'LRc', 'MLPmc', 'NBmc', 'DTmc']

a = [0.7754, 0.7618, 0.7510, 0.7775, 0.7667, 0.7855, 0.7949, 0.8011]
b = [0.7902, 0.7270, 0.6873, 0.6263, 0.6036, 0.7637, 0.6939, 0.7027]

l1, = plt.plot(indexes, a, color='red', label='Precision')
l2, = plt.plot(indexes, b, color='blue', label='Recall')

plt.legend(handler_map={l1: HandlerLine2D(numpoints=2)})

plt.ylabel('Metric Value')
plt.xlabel('Classifiers')

plt.axis([0, 7, 0.6, 0.82])
plt.show()