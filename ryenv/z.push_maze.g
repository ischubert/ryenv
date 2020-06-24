floor 	{  shape:ssBox, size:[1., 1., 0.1, 0.02], contact:-1 X:<[0.5, 0.5, 0., 1, 0, 0, 0]> 
    fixed, contact, logical:{ }
    friction:1}
disk(floor) 	{  shape:cylinder, size:[0.1, 0.1], contact:-1 X:<[0., 0., 0.1, 1, 0, 0, 0]>
    mass:1
    joint:rigid
    friction:.1}
finger 	{  shape:sphere, size:[0.02], contact:-1 X:<[0.5, 0.5, 0.1, 1, 0, 0, 0]> 
    friction:.1}
