# fail to trigger a shape error by passing coords, box, or params of wrong shape to unbound_impl.execute
# (runs silently but returns garbage answers, sometimes of the wrong shape, sometimes also causing a core dump)

from timemachine.lib import potentials
import numpy as np


harmonic_bond = potentials.HarmonicBond(np.array([[0, 1], [0, 2], [999, 1000]], dtype=np.int32)).bind([[1, 0], [1, 0]])

unbound_impl = harmonic_bond.unbound_impl(np.float32)

bad_coords_shapes = [
    (10, 3),  # too few particles
    (10,),  # too few dimensions
    (10, 1),
    (10, 2),
    (10, 3, 3),  # too many dimensions
]

bad_params_shapes = [
    (2, 2),  # too few bonds
    (3,),  # too few dimensions
    (3, 1),
    (3, 2, 2),  # too many dimensions
]

bad_box_shapes = [
    (3,),
    (4, 4),
    (2, 2),
    (3, 3, 3),
]

results = []
for coords_shape in bad_coords_shapes:
    for params_shape in bad_params_shapes:
        for box_shape in bad_box_shapes:
            res = unbound_impl.execute(
                coords=np.ones(coords_shape),
                params=np.ones(params_shape),
                box=np.ones(box_shape),
                lam=1.0,
            )
            results.append(res)

for (du_dx, du_dp, u, du_dl) in results:
    print(du_dx.shape, du_dp.shape, u, du_dl)
