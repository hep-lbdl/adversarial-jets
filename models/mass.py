
import keras.backend as K


def image_mass_func(eta_grid=None, phi_grid=None, shape=(None, 1, 25, 25), use_keras=True):
	"""
	returns a Keras Backend enabled function that calculates the mass.

	If eta_grid and phi_grid are both None, then uses the default eta and phi
	grids, and assumes that images are 25 x 25, eta and phi going
	from -1.25 to 1.25, where the grids here are the midpoints of this
	mesh, i.e., -1.2 to 1.2 in intervals of 0.1

	Example:

	>>> jet_mass = image_mass_func()
	>>> minibatch = np.random.uniform(0, 1, (100, 1, 25, 25))
	>>> mass = jet_mass(minibatch)
	"""

	if not use_keras:
		import numpy as K

	if eta_grid is None and phi_grid is None:
		eta_grid = np.tile(np.linspace(-1.2, 1.2, 25), (25, 1))
		phi_grid = np.tile(np.linspace(-1.2, 1.2, 25).reshape(-1, 1), (1, 25))

	if len(shape) > 2:
		axis_slice = tuple(range(1, len(shape[1:]) + 1))
	if len(shape) == 2:
		axis_slice = None
	else:
		raise ValueError('invalid shape')


    def _(jet_image):
        Px = K.sum(jet_image * K.cos(phi_grid), axis=axis_slice)
        Py = K.sum(jet_image * K.sin(phi_grid), axis=axis_slice)

        Pz = K.sum(jet_image * K.sinh(eta_grid), axis=axis_slice)
        E = K.sum(jet_image * K.cosh(eta_grid), axis=axis_slice)

        PT = K.sqrt(K.square(Px) + K.square(Py))
        M2 = K.square(E) - (K.square(PT) + K.square(Pz))
        M = K.sqrt(M2)
    return _
