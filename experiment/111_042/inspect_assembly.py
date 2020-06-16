import numpy as np
import matplotlib.pyplot as plt

for subset in (10, 0, 1):
	dct = np.load('assembled_%u.npz'%subset)
	W = dct['W']
	Pjlk = dct['Pjlk']

	fig, ax = plt.subplots(ncols=4, figsize=(12,3))

	Nj = Pjlk.shape[0]
	[a.clear() for a in ax]
	#ax[0].imshow(np.abs(W[:,64,:]), vmax=np.abs(W[:,64,:]).max()/2)
	#ax[1].imshow(np.abs(W[Nj//2,:,:]), vmax=np.abs(W[Nj//2]).max()/10)
	ax[0].imshow(np.log10(np.abs(W[:,64,:])), vmin=-3)
	ax[1].imshow(np.log10(np.abs(W[Nj//2,:,:])), vmin=-3)
	Pjk = np.sum(Pjlk, axis=1)
	ax[2].imshow(np.abs(Pjk), vmax=np.abs(Pjk).max()/2, aspect='auto')
	Plk = np.sum(Pjlk, axis=0)
	if Plk.shape[0] > 1:
		ax[3].imshow(np.abs(Plk), vmax=np.abs(Plk).max()/2, aspect='auto')
	ax[0].set_title('model from above')
	ax[1].set_title('central model slice')
	ax[2].set_title('|Pjk|')
	ax[3].set_title('|Plk|')
	fig.suptitle('subset %u'%subset)
	plt.draw()
	plt.pause(.01)

plt.show()
