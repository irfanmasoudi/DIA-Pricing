import numpy as np
from Learner import *

Alessandro, [2 Apr 2021 12.19.29]:
[(ip.dst == 18.185.199.22 || ip.dst == 3.120.68.56) && mqtt.msgtype == 1 && mqtt.conflag.willflag == 1]

Simone, [2 Apr 2021 12.19.38]:
mqtt.msgtype == 1 && mqtt.willmsg && (ip.dst == 3.120.68.56 || ip.dst == 18.185.199.22)


mqtt.msgtype == 1 && mqtt.willmsg && ip.dst == 18.185.199.22