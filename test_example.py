from main import in_hand
import numpy as np

# def test_example_1():
#     positions = np.array(
#         [[[700, 700, 200],[700,-700,200],[-700,700,200],[-700,-700,200]],
#          [[700, 700, 200],[700,-700,200],[-700,700,200],[-700,-700,200]],
#          [[500, 700, 400],[700,-700,200],[-700,700,200],[-700,-700,200]],
#          [[500, 680, 300],[700,-700,200],[-700,700,200],[-700,-700,200]],
#          [[510, 650, 280],[700,-700,200],[-700,700,200],[-700,-700,200]],
#          [[505, 620, 260],[700,-700,200],[-700,700,200],[-700,-700,200]],
#          [[800, 150, 205],[700,-700,200],[-700,700,200],[-700,-700,200]],
#          [[800, 150, 200],[930,-684,610],[-700,700,200],[-700,-700,200]],
#          [[800, 150, 200],[930,-684,612],[-700,700,200],[-700,-700,200]],
#          [[800, 150, 200],[934,-674,623],[-700,700,200],[-700,-700,200]],
#          [[800, 150, 200],[914,-681,200],[-700,700,200],[-700,-700,200]],
#          [[800, 150, 200],[914,-681,200],[-12,654,400],[-700,-700,200]],
#          [[800, 150, 200],[914,-681,200],[-215,634,370],[-700,-700,200]],
#          [[800, 150, 200],[914,-681,200],[-215,140,200],[-700,-700,300]],
#          [[800, 150, 200],[914,-681,200],[-215,140,200],[-650,-700,500]],
#          [[800, 150, 200],[914,-681,200],[-215,140,200],[0,-650,350]],
#          [[800, 150, 200],[860,-681,200],[-215,140,200],[40,-650,250]],
#          [[800, 150, 200],[855,-671,550],[-215,140,200],[40,-650,250]],
#          [[800, 150, 200],[840,-661,550],[-215,140,200],[40,-650,250]],
#          [[800, 150, 200],[873,-671,550],[-215,140,200],[40,-650,250]],
#          [[800, 150, 200],[850,-661,550],[-215,140,200],[40,-650,250]],
#          [[780, 160, 540],[860,-671,550],[-215,140,200],[40,-650,250]],
#          [[795, 158, 540],[870,-661,550],[-215,140,200],[40,-650,250]],
#          [[-450, -640, 283],[810,-550,230],[-215,140,200],[40,-650,250]],
#          [[-400, -650, 200],[810,-550,200],[-215,140,200],[40,-650,250]],
#          [[-360, -650, 200],[810,-550,200],[100,200,800],[40,-650,250]],
#          [[-360, -650, 210],[810,-550,200],[450,-630,200],[40,-650,250]],
#          [[-360, -650, 215],[810,-550,200],[440,-630,215],[40,-650,250]],
#          [[-360, -650, 215],[750,-530,800],[440,-630,215],[40,-650,250]],
#          [[-360, -650, 215],[450,-625,605],[440,-630,215],[40,-650,250]],
#          [[-360, -650, 220],[450,-631,705],[440,-630,205],[40,-650,250]],
#          [[-360, -650, 215],[60,-631,650],[440,-630,215],[40,-650,250]],
#          [[-360, -650, 220],[41,-650,650],[440,-630,205],[40,-650,250]],
#          [[-360, -650, 220],[41,-650,650],[440,-630,205],[40,-650,250]],
#          [[-360, -650, 220],[41,-650,650],[440,-630,205],[40,-650,250]],
#          [[-360, -650, 220],[41,-650,650],[440,-630,205],[40,-650,250]],
#          [[-60, -650, 220],[-259,-650,650],[140,-630,205],[-259,-650,250]],
#          [[-360, -650, 220],[-559,-650,650],[-160,-630,205],[-559,-650,250]]])

#     connections = np.array([[[False,False,False,False,False,False],[False,False,False,False,False,False],[False,False,False,False,False,False],[False,False,False,False,False,False]],[[False,False,False,False,False,False],[False,False,False,False,False,False],[False,False,False,False,False,False],[False,False,False,False,False,False]],[[False,False,False,False,False,False],[False,False,False,False,False,False],[False,False,False,False,False,False],[False,False,False,False,False,False]],[[False,False,False,False,False,False],[False,False,False,False,False,False],[False,False,False,False,False,False],[False,False,False,False,False,False]],[[False,False,False,False,False,False],[False,False,False,False,False,False],[False,False,False,False,False,False],[False,False,False,False,False,False]],[[False,False,False,False,False,False],[False,False,False,False,False,False],[False,False,False,False,False,False],[False,False,False,False,False,False]],[[False,False,False,False,False,False],[False,False,False,False,False,False],[False,False,False,False,False,False],[False,False,False,False,False,False]],[[False,False,False,False,False,False],[False,False,False,False,False,False],[False,False,False,False,False,False],[False,False,False,False,False,False]],[[False,False,False,False,False,False],[False,False,False,False,False,False],[False,False,False,False,False,False],[False,False,False,False,False,False]],[[False,False,False,False,False,False],[False,False,False,False,False,False],[False,False,False,False,False,False],[False,False,False,False,False,False]],[[False,False,False,False,False,False],[False,False,False,False,False,False],[False,False,False,False,False,False],[False,False,False,False,False,False]],[[False,False,False,False,False,False],[False,False,False,False,False,False],[False,False,False,False,False,False],[False,False,False,False,False,False]],[[False,False,False,False,False,False],[False,False,False,False,False,False],[False,False,False,False,False,False],[False,False,False,False,False,False]],[[False,False,False,False,False,False],[False,False,False,False,False,False],[False,False,False,False,False,False],[False,False,False,False,False,False]],[[False,False,False,False,False,False],[False,False,False,False,False,False],[False,False,False,False,False,False],[False,False,False,False,False,False]],[[False,False,False,False,False,False],[False,False,False,False,False,False],[False,False,False,False,False,False],[False,False,False,False,False,False]],[[False,False,False,False,False,False],[False,False,False,False,False,False],[False,False,False,False,False,False],[False,False,False,False,False,False]],[[False,False,False,False,False,False],[False,False,False,False,False,False],[False,False,False,False,False,False],[False,False,False,False,False,False]],[[False,False,False,False,False,False],[False,False,False,False,False,False],[False,False,False,False,False,False],[False,False,False,False,False,False]],[[False,False,False,False,False,False],[False,False,False,False,False,False],[False,False,False,False,False,False],[False,False,False,False,False,False]],[[False,False,False,False,False,False],[False,False,False,False,False,False],[False,False,False,False,False,False],[False,False,False,False,False,False]],[[False,False,False,False,False,False],[False,False,False,False,False,False],[False,False,False,False,False,False],[False,False,False,False,False,False]],[[False,False,False,False,False,False],[False,False,False,False,False,False],[False,False,False,False,False,False],[False,False,False,False,False,False]],[[False,False,False,False,False,False],[False,False,False,False,False,False],[False,False,False,False,False,False],[False,False,False,False,False,False]],[[False,False,False,False,False,False],[False,False,False,False,False,False],[False,False,False,False,False,False],[False,False,False,False,False,False]],[[False,True,False,False,False,False],[False,False,False,False,False,False],[False,False,False,False,False,False],[False,False,True,False,False,False]],[[False,True,False,False,False,False],[False,False,False,False,False,False],[False,False,False,False,False,False],[False,False,True,False,False,False]],[[False,True,False,False,False,False],[False,False,False,False,False,False],[False,False,False,False,True,False],[False,True,True,False,False,False]],[[False,True,False,False,False,False],[False,False,False,False,False,False],[False,False,False,False,True,False],[False,True,True,False,False,False]],[[False,True,False,False,False,False],[False,False,False,True,False,False],[False,False,True,False,True,False],[False,True,True,False,False,False]],[[False,True,False,False,False,False],[False,False,False,False,False,False],[False,False,False,False,True,False],[False,True,True,False,False,False]],[[False,True,False,False,False,False],[False,False,False,False,False,False],[False,False,False,False,True,False],[False,True,True,False,False,False]],[[False,True,False,False,False,False],[False,False,False,True,False,False],[False,False,False,False,True,False],[True,True,True,False,False,False]],[[False,True,False,False,False,False],[False,False,False,True,False,False],[False,False,False,False,True,False],[True,True,True,False,False,False]],[[False,True,False,False,False,False],[False,False,False,True,False,False],[False,False,False,False,True,False],[True,True,True,False,False,False]],[[False,True,False,False,False,False],[False,False,False,True,False,False],[False,False,False,False,True,False],[True,True,True,False,False,False]],[[False,True,False,False,False,False],[False,False,False,True,False,False],[False,False,False,False,True,False],[True,True,True,False,False,False]],[[False,True,False,False,False,False],[False,False,False,True,False,False],[False,False,False,False,True,False],[True,True,True,False,False,False]]])

#     expected = np.array(
#         [[False, False, False, False], [False, False, False, False], [True, False, False, False],
#          [True, False, False, False], [True, False, False, False], [True, False, False, False],
#          [True, False, False, False], [False, True, False, False], [False, True, False, False],
#          [False, True, False, False], [False, False, False, False], [False, False, True, False],
#          [False, False, True, False], [False, False, False, True], [False, False, False, True],
#          [False, False, False, True], [False, True, False, False], [False, True, False, False],
#          [False, True, False, False], [False, True, False, False], [False, True, False, False],
#          [True, True, False, False], [True, True, False, False], [True, True, False, False],
#          [True, False, False, False], [True, False, True, False], [False, False, True, False],
#          [False, False, False, False], [False, True, False, False], [False, True, False, False],
#          [False, True, False, False], [False, True, False, False], [False, False, False, False],
#          [False, False, False, False], [False, False, False, False], [False, False, False, False],
#          [False, False, False, False], [False, False, False, False]])

#     assert (in_hand(positions, connections) == expected).all()

#####################################################################
#################### Results obtained ##############################

# [[False, False, False, False],
#  [False, False, False, False],
#  [ True, False, False, False],
#  [ True, False, False, False],
#  [ True, False, False, False],
#  [ True, False, False, False],
#  [ True, False, False, False],
#  [False,  True, False, False],
#  [False,  True, False, False],
#  [False,  True, False, False],
#  [False,  True, False, False],
#  [False, False,  True, False],
#  [False, False,  True, False],
#  [False, False,  True,  True],
#  [False, False, False,  True],
#  [False, False, False,  True],
#  [False, False, False,  True],
#  [False,  True, False, False],
#  [False,  True, False, False],
#  [False,  True, False, False],
#  [False,  True, False, False],
#  [ True,  True, False, False],
#  [ True,  True, False, False],
#  [ True,  True, False, False],
#  [ True, False, False, False],
#  [False, False,  True, False],
#  [False, False,  True, False],
#  [False, False, False, False],
#  [False,  True, False, False],
#  [False, False, False, False],
#  [False,  True, False, False],
#  [False,  True, False, False],
#  [False, False, False, False],
#  [False, False, False, False],
#  [False, False, False, False],
#  [False, False, False, False],
#  [False, False, False, False],
#  [False, False, False, False]]
