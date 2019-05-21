import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D

# FEATURES

import numpy as np
indexes = np.linspace(5, 240, 48)

a = [0.59418490619737951, 0.59098151200242022, 0.55787337897615263, 0.55347025869847843, 0.54922741538577491, 0.54714326901304933, 0.54654120560214059, 0.54685534059355856, 0.54820553101647329, 0.54978842842942088, 0.55756418782280248, 0.55951495154589748, 0.55956255048585901, 0.5601573352789776, 0.56197689420658414, 0.5624171905417934, 0.56426496195961551, 0.56650548808463663, 0.5680901621680684, 0.56703186969954344, 0.56729078326150328, 0.56851481234062662, 0.56879878623549129, 0.56921597084413322, 0.56845798617653742, 0.56916183834174605, 0.57014863278669958, 0.5703036370521688, 0.57077294824490921, 0.57069306766620387, 0.57061404676676497, 0.57058720976337574, 0.57074295908420947, 0.57095163735645693, 0.57121295802036309, 0.57126514191641276, 0.57128934257005115, 0.57097600994594877, 0.57066273463379735, 0.57081934363389752, 0.5704277924776715, 0.56982733380139361, 0.57011486103722686, 0.57029756198535198, 0.5704283082852315, 0.57048072142908568, 0.57066342237721057, 0.57084635257314009]
b = [0.59269921545743365, 0.59156624109211153, 0.55792352625653541, 0.55287815423246278, 0.54899552321503298, 0.54788045304238653, 0.54725925860432767, 0.54796525849822864, 0.54900841718095172, 0.55209714829681766, 0.5611419338296405, 0.56273282818458858, 0.56344759597531435, 0.56612983560561847, 0.56811896930794026, 0.56797514632809321, 0.5694786564340153, 0.56996685882359877, 0.57046692366182739, 0.56909205849587108, 0.5693320336215687, 0.57051540497368158, 0.57232770714340442, 0.57115929279175726, 0.57115361944675302, 0.57178719209209383, 0.57180524364438001, 0.57196557090060307, 0.57236123145479301, 0.57189572244522646, 0.57189417516931618, 0.57158228855369486, 0.57236793631707072, 0.57299789865195427, 0.57276204959535049, 0.57244861570381889, 0.5722205030267663, 0.57206327032236393, 0.57230221393078817, 0.57238083028298947, 0.57214498122638568, 0.572144465467749, 0.5721434339504754, 0.57237979876571587, 0.57237928300707919, 0.57245738360064369, 0.5722220503026767, 0.5721429181918386]

l1, = plt.plot(indexes, a, label='Train AUC')
l2, = plt.plot(indexes, b, color='red', label='Test AUC')

plt.legend(handler_map={l1: HandlerLine2D(numpoints=2)})

plt.ylabel('AUC')
plt.xlabel('Features')

plt.axis([5, 240, 0.5, 0.6])
plt.show()