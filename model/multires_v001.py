try:
	import torch
	from torch.nn import Module
except:
	torch = object
	Module = object

class Conv2d_batchnorm(Module):
	'''
	2D Convolutional layers

	Arguments:
		num_in_filters {int} -- number of input filters
		num_out_filters {int} -- number of output filters
		kernel_size {tuple} -- size of the convolving kernel
		stride {tuple} -- stride of the convolution (default: {(1, 1)})
		activation {str} -- activation function (default: {'relu'})

	'''
	def __init__(self, num_in_filters, num_out_filters, kernel_size, stride = (1,1), activation = True, inplace = False):
		super().__init__()
		self.activation = activation
		self.conv1 = torch.nn.Conv2d(
			in_channels=num_in_filters,
			out_channels=num_out_filters,
			kernel_size=kernel_size,
			stride=stride,
			padding = 'same',
			padding_mode = 'reflect',
			# bias=False
			)
		self.act = torch.nn.SELU(inplace = inplace)
	
	def forward(self,x):
		x = self.conv1(x)
		
		if self.activation:
			return self.act(x)
		else:
			return x
	
	'''

	def __init__(self, num_in_filters, num_out_filters, kernel_size, stride = (1,1), activation = True, inplace = False):
		super().__init__()
		layers = [
			torch.nn.Conv2d(
				in_channels=num_in_filters, 
				out_channels=num_out_filters, 
				kernel_size=kernel_size, 
				stride=stride, 
				padding = 'same',
				padding_mode = 'reflect',
				bias=False
			),
			torch.nn.BatchNorm2d(num_out_filters),
		]

		if activation:
			# layers.insert(2, torch.nn.ELU(inplace=True))
			layers.append(torch.nn.SELU(inplace = inplace))
		
		self.layers = torch.nn.Sequential(*layers)
	
	def forward(self,x):
		return self.layers(x)
	'''

class Multiresblock(Module):
	'''
	MultiRes Block
	
	Arguments:
		num_in_channels {int} -- Number of channels coming into mutlires block
		num_filters {int} -- Number of filters in a corrsponding UNet stage
		alpha {float} -- alpha hyperparameter (default: 1.67)
	
	'''

	def __init__(self, num_in_channels, num_filters, alpha=1.69):
	
		super().__init__()
		self.alpha = alpha
		self.W = num_filters * alpha
		
		filt_cnt_3x3 = int(self.W*0.167)
		filt_cnt_5x5 = int(self.W*0.333)
		filt_cnt_7x7 = int(self.W*0.5)
		num_out_filters = filt_cnt_3x3 + filt_cnt_5x5 + filt_cnt_7x7
		
		self.shortcut = Conv2d_batchnorm(num_in_channels ,num_out_filters , kernel_size = (1,1), activation=False)

		self.conv_3x3 = Conv2d_batchnorm(num_in_channels, filt_cnt_3x3, kernel_size = (3,3), activation=True)

		self.conv_5x5 = Conv2d_batchnorm(filt_cnt_3x3, filt_cnt_5x5, kernel_size = (3,3), activation=True)
		
		self.conv_7x7 = Conv2d_batchnorm(filt_cnt_5x5, filt_cnt_7x7, kernel_size = (3,3), activation=True)

		self.act = torch.nn.SELU()

	def forward(self,x):

		shrtct = self.shortcut(x)
		
		a = self.conv_3x3(x)
		b = self.conv_5x5(a)
		c = self.conv_7x7(b)

		x = torch.cat([a,b,c],axis=1)

		x = x + shrtct
		x = self.act(x)
	
		return x

class Multiresblock_MemOpt(Module):
	'''
	MultiRes Block
	
	Arguments:
		num_in_channels {int} -- Number of channels coming into mutlires block
		num_filters {int} -- Number of filters in a corrsponding UNet stage
		alpha {float} -- alpha hyperparameter (default: 1.67)
	
	'''

	def __init__(self, num_in_channels, num_filters, alpha=1.69):
		super().__init__()
		self.alpha = alpha
		self.W = num_filters * alpha
		
		filt_cnt_3x3 = int(self.W*0.167)
		filt_cnt_5x5 = int(self.W*0.333)
		filt_cnt_7x7 = int(self.W*0.5)
		num_out_filters = filt_cnt_3x3 + filt_cnt_5x5 + filt_cnt_7x7
		
		self.shortcut = Conv2d_batchnorm(num_in_channels ,num_out_filters , kernel_size = (1,1), activation=False)

		self.conv_3x3 = Conv2d_batchnorm(num_in_channels, filt_cnt_3x3, kernel_size = (3,3), activation=True, inplace=True)

		self.conv_5x5 = Conv2d_batchnorm(filt_cnt_3x3, filt_cnt_5x5, kernel_size = (3,3), activation=True, inplace=True)
		
		self.conv_7x7 = Conv2d_batchnorm(filt_cnt_5x5, filt_cnt_7x7, kernel_size = (3,3), activation=True, inplace=True)

		self.act = torch.nn.SELU(inplace=True)

	def forward(self,x):
		shrtct = self.shortcut(x)
		a = self.conv_3x3(x)
		del x
		b = self.conv_5x5(a)
		c = self.conv_7x7(b)
		x = torch.cat([a,b,c],axis=1)
		del a, b, c
		x = x + shrtct
		x = self.act(x)
		return x
		'''
		except:
			print ('flameSimpleML Multiresblock: Low GPU memory - trying mixed mode (slow)')
			x_device = x.device
			x_dtype = x.dtype
			shrtct = self.shortcut(x)
			shrtct_cpu = shrtct.cpu()
			print ('Multiresblock shrtct passed')
			del shrtct
			a = self.conv_3x3(x)
			print ('Multiresblock a conv passed')
			del x
			b = self.conv_5x5(a)
			a_cpu = a.cpu()
			print ('Multiresblock b conv passed')
			del a
			c = self.conv_7x7(b)
			print ('Multiresblock c conv passed')
			b_cpu = b.cpu()
			c_cpu = c.cpu()
			del b, c
			x_cpu = torch.cat([a_cpu, b_cpu, c_cpu], axis=1)
			del a_cpu, b_cpu, c_cpu
			x_cpu = x_cpu + shrtct_cpu
			x_cpu = self.act(x_cpu)
			try:
				x = x_cpu.to(device=x_device, dtype=x_dtype)
				del x_cpu
				return x
			except:
				return x_cpu
			'''

class Respath(Module):
	'''
	ResPath
	
	Arguments:
		num_in_filters {int} -- Number of filters going in the respath
		num_out_filters {int} -- Number of filters going out the respath
		respath_length {int} -- length of ResPath
		
	'''

	def __init__(self, num_in_filters, num_out_filters, respath_length):
	
		super().__init__()

		self.respath_length = respath_length
		self.shortcuts = torch.nn.ModuleList([])
		self.convs = torch.nn.ModuleList([])
		self.bns = torch.nn.ModuleList([])
		self.act = torch.nn.SELU(inplace=True)

		for i in range(self.respath_length):
			if(i==0):
				self.shortcuts.append(Conv2d_batchnorm(num_in_filters, num_out_filters, kernel_size = (1,1), activation=False))
				self.convs.append(Conv2d_batchnorm(num_in_filters, num_out_filters, kernel_size = (3,3),activation=True))
			else:
				self.shortcuts.append(Conv2d_batchnorm(num_out_filters, num_out_filters, kernel_size = (1,1), activation=False))
				self.convs.append(Conv2d_batchnorm(num_out_filters, num_out_filters, kernel_size = (3,3), activation=True))
		
	def forward(self,x):
		for i in range(self.respath_length):
			shortcut = self.shortcuts[i](x)
			x = self.convs[i](x)
			x = x + shortcut
			x = self.act(x)

		return x
	
class Respath4(Module):
	def __init__(self, num_in_filters, num_out_filters, respath_length):
	
		super().__init__()
		self.act = torch.nn.SELU()
		self.shortcut1 = Conv2d_batchnorm(num_in_filters, num_out_filters, kernel_size = (1,1), activation=False)
		self.shortcut2 = Conv2d_batchnorm(num_out_filters, num_out_filters, kernel_size = (1,1), activation=False)
		self.shortcut3 = Conv2d_batchnorm(num_out_filters, num_out_filters, kernel_size = (1,1), activation=False)
		self.shortcut4 = Conv2d_batchnorm(num_out_filters, num_out_filters, kernel_size = (1,1), activation=False)
		self.conv1 = Conv2d_batchnorm(num_in_filters, num_out_filters, kernel_size = (3,3),activation=True)
		self.conv2 = Conv2d_batchnorm(num_out_filters, num_out_filters, kernel_size = (3,3), activation=True)
		self.conv3 = Conv2d_batchnorm(num_out_filters, num_out_filters, kernel_size = (3,3), activation=True)
		self.conv4 = Conv2d_batchnorm(num_out_filters, num_out_filters, kernel_size = (3,3), activation=True)
		
	def forward(self,x):
		shortcut = self.shortcut1(x)
		x = self.conv1(x)
		x = x + shortcut
		x = self.act(x)

		shortcut = self.shortcut2(x)
		x = self.conv2(x)
		x = x + shortcut
		x = self.act(x)
		
		shortcut = self.shortcut3(x)
		x = self.conv3(x)
		x = x + shortcut
		x = self.act(x)

		shortcut = self.shortcut4(x)
		x = self.conv4(x)
		x = x + shortcut
		x = self.act(x)

		return x

class Respath3(Module):
	def __init__(self, num_in_filters, num_out_filters, respath_length):
	
		super().__init__()
		self.act = torch.nn.SELU()
		self.shortcut1 = Conv2d_batchnorm(num_in_filters, num_out_filters, kernel_size = (1,1), activation=False)
		self.shortcut2 = Conv2d_batchnorm(num_out_filters, num_out_filters, kernel_size = (1,1), activation=False)
		self.shortcut3 = Conv2d_batchnorm(num_out_filters, num_out_filters, kernel_size = (1,1), activation=False)
		self.conv1 = Conv2d_batchnorm(num_in_filters, num_out_filters, kernel_size = (3,3),activation=True)
		self.conv2 = Conv2d_batchnorm(num_out_filters, num_out_filters, kernel_size = (3,3), activation=True)
		self.conv3 = Conv2d_batchnorm(num_out_filters, num_out_filters, kernel_size = (3,3), activation=True)
		
	def forward(self,x):
		shortcut = self.shortcut1(x)
		x = self.conv1(x)
		x = x + shortcut
		x = self.act(x)

		shortcut = self.shortcut2(x)
		x = self.conv2(x)
		x = x + shortcut
		x = self.act(x)

		shortcut = self.shortcut3(x)
		x = self.conv3(x)
		x = x + shortcut
		x = self.act(x)

		return x

class Respath2(Module):
	def __init__(self, num_in_filters, num_out_filters, respath_length):
	
		super().__init__()
		self.act = torch.nn.SELU()
		self.shortcut1 = Conv2d_batchnorm(num_in_filters, num_out_filters, kernel_size = (1,1), activation=False)
		self.shortcut2 = Conv2d_batchnorm(num_out_filters, num_out_filters, kernel_size = (1,1), activation=False)
		self.conv1 = Conv2d_batchnorm(num_in_filters, num_out_filters, kernel_size = (3,3),activation=True)
		self.conv2 = Conv2d_batchnorm(num_out_filters, num_out_filters, kernel_size = (3,3), activation=True)
		
	def forward(self,x):
		shortcut = self.shortcut1(x)
		x = self.conv1(x)
		x = x + shortcut
		x = self.act(x)

		shortcut = self.shortcut2(x)
		x = self.conv2(x)
		x = x + shortcut
		x = self.act(x)

		return x

class Respath1(Module):
	def __init__(self, num_in_filters, num_out_filters, respath_length):
	
		super().__init__()
		self.act = torch.nn.SELU()
		self.shortcut1 = Conv2d_batchnorm(num_in_filters, num_out_filters, kernel_size = (1,1), activation=False)
		self.conv1 = Conv2d_batchnorm(num_in_filters, num_out_filters, kernel_size = (3,3),activation=True)
		
	def forward(self,x):
		shortcut = self.shortcut1(x)
		x = self.conv1(x)
		x = x + shortcut
		x = self.act(x)

		return x

class Respath4_MemOpt(Module):
	def __init__(self, num_in_filters, num_out_filters, respath_length):
	
		super().__init__()
		self.act = torch.nn.SELU(inplace=True)
		self.shortcut1 = Conv2d_batchnorm(num_in_filters, num_out_filters, kernel_size = (1,1), activation=False)
		self.shortcut2 = Conv2d_batchnorm(num_out_filters, num_out_filters, kernel_size = (1,1), activation=False)
		self.shortcut3 = Conv2d_batchnorm(num_out_filters, num_out_filters, kernel_size = (1,1), activation=False)
		self.shortcut4 = Conv2d_batchnorm(num_out_filters, num_out_filters, kernel_size = (1,1), activation=False)
		self.conv1 = Conv2d_batchnorm(num_in_filters, num_out_filters, kernel_size = (3,3),activation=True)
		self.conv2 = Conv2d_batchnorm(num_out_filters, num_out_filters, kernel_size = (3,3), activation=True)
		self.conv3 = Conv2d_batchnorm(num_out_filters, num_out_filters, kernel_size = (3,3), activation=True)
		self.conv4 = Conv2d_batchnorm(num_out_filters, num_out_filters, kernel_size = (3,3), activation=True)

		
	def forward(self,x):
		# x_device = x.device
		# x_dtype = x.dtype
		# try:
		shortcut = self.shortcut1(x)
		x = self.conv1(x)
		x = x + shortcut
		x = self.act(x)
		del shortcut
		'''
		except:
			try:
				shortcut = self.shortcut1(x)
				shortcut_cpu = shortcut.cpu()
				del shortcut
			except:
				print ('ResPath4 step 01 shortcut on CPU')
				shortcut1cpu  = self.shortcut1.to(device='cpu', dtype=torch.float32)
				shortcut_cpu = shortcut1cpu(x.to(device='cpu', dtype=torch.float32))
				del shortcut1cpu
				shortcut_cpu.to(dtype = x_dtype)
			try:
				x = self.conv1(x)
				x_cpu = x.cpu()
				del x
			except:
				print ('ResPath4 step 01 conv on CPU')
				conv1cpu = self.conv1.to(device='cpu', dtype=torch.float32)
				x_cpu = conv1cpu(x.to(device='cpu', dtype=torch.float32))
				del conv1cpu
				x_cpu.to(dtype=x_dtype)
			x_cpu = x_cpu + shortcut_cpu
			del shortcut_cpu
			x_cpu = self.act(x_cpu)
			x = x_cpu.to(device=x_device, dtype=x_dtype)
			del x_cpu
	
		try:
			shortcut = self.shortcut2(x)
			x = self.conv2(x)
			x = x + shortcut
			x = self.act(x)
			del shortcut
		except:
			try:
				shortcut = self.shortcut2(x)
				shortcut_cpu = shortcut.cpu()
				del shortcut
			except:
				print ('ResPath4 step 02 shortcut on CPU')
				shortcut2cpu  = self.shortcut2.to(device='cpu', dtype=torch.float32)
				shortcut_cpu = shortcut2cpu(x.to(device='cpu', dtype=torch.float32))
				del shortcut2cpu
				shortcut_cpu.to(dtype = x_dtype)
			try:
				x = self.conv2(x)
				x_cpu = x.cpu()
				del x
			except:
				print ('ResPath4 step 02 conv on CPU')
				conv2cpu = self.conv2.to(device='cpu', dtype=torch.float32)
				x_cpu = conv2cpu(x.to(device='cpu', dtype=torch.float32))
				del conv2cpu
				x_cpu.to(dtype=x_dtype)
			x_cpu = x_cpu + shortcut_cpu
			del shortcut_cpu
			x_cpu = self.act(x_cpu)
			x = x_cpu.to(device=x_device, dtype=x_dtype)
			del x_cpu

		try:
			shortcut = self.shortcut3(x)
			x = self.conv3(x)
			x = x + shortcut
			x = self.act(x)
			del shortcut
		except:
			try:
				shortcut = self.shortcut3(x)
				shortcut_cpu = shortcut.cpu()
				del shortcut
			except:
				print ('ResPath4 step 03 shortcut on CPU')
				shortcut3cpu  = self.shortcut3.to(device='cpu', dtype=torch.float32)
				shortcut_cpu = shortcut3cpu(x.to(device='cpu', dtype=torch.float32))
				del shortcut3cpu
				shortcut_cpu.to(dtype = x_dtype)
			try:
				x = self.conv3(x)
				x_cpu = x.cpu()
				del x
			except:
				print ('ResPath4 step 03 conv on CPU')
				conv3cpu = self.conv3.to(device='cpu', dtype=torch.float32)
				x_cpu = conv3cpu(x.to(device='cpu', dtype=torch.float32))
				del conv3cpu
				x_cpu.to(dtype=x_dtype)
			x_cpu = x_cpu + shortcut_cpu
			del shortcut_cpu
			x_cpu = self.act(x_cpu)
			x = x_cpu.to(device=x_device, dtype=x_dtype)
			del x_cpu

		try:
			shortcut = self.shortcut4(x)
			x = self.conv4(x)
			x = x + shortcut
			x = self.act(x)
			del shortcut
		except:
			try:
				shortcut = self.shortcut3(x)
				shortcut_cpu = shortcut.cpu()
				del shortcut
			except:
				print ('ResPath4 step 04 shortcut on CPU')
				shortcut4cpu  = self.shortcut4.to(device='cpu', dtype=torch.float32)
				shortcut_cpu = shortcut4cpu(x.to(device='cpu', dtype=torch.float32))
				del shortcut4cpu
				shortcut_cpu.to(dtype = x_dtype)
			try:
				x = self.conv4(x)
				x_cpu = x.cpu()
				del x
			except:
				print ('ResPath4 step 04 conv on CPU')
				conv4cpu = self.conv4.to(device='cpu', dtype=torch.float32)
				x_cpu = conv4cpu(x.to(device='cpu', dtype=torch.float32))
				del conv4cpu
				x_cpu.to(dtype=x_dtype)
			x_cpu = x_cpu + shortcut_cpu
			del shortcut_cpu
			x_cpu = self.act(x_cpu)
			print (f'x_cpu shape: {x_cpu.shape}')
			x = x_cpu.to(device=x_device, dtype=x_dtype)
			del x_cpu
		'''

		return x

class Respath3_MemOpt(Module):
	def __init__(self, num_in_filters, num_out_filters, respath_length):
	
		super().__init__()
		self.act = torch.nn.SELU(inplace=True)
		self.shortcut1 = Conv2d_batchnorm(num_in_filters, num_out_filters, kernel_size = (1,1), activation=False)
		self.shortcut2 = Conv2d_batchnorm(num_out_filters, num_out_filters, kernel_size = (1,1), activation=False)
		self.shortcut3 = Conv2d_batchnorm(num_out_filters, num_out_filters, kernel_size = (1,1), activation=False)
		self.conv1 = Conv2d_batchnorm(num_in_filters, num_out_filters, kernel_size = (3,3),activation=True)
		self.conv2 = Conv2d_batchnorm(num_out_filters, num_out_filters, kernel_size = (3,3), activation=True)
		self.conv3 = Conv2d_batchnorm(num_out_filters, num_out_filters, kernel_size = (3,3), activation=True)
		
	def forward(self,x):
		shortcut = self.shortcut1(x)
		x = self.conv1(x)
		x = x + shortcut
		x = self.act(x)

		shortcut = self.shortcut2(x)
		x = self.conv2(x)
		x = x + shortcut
		x = self.act(x)

		shortcut = self.shortcut3(x)
		x = self.conv3(x)
		x = x + shortcut
		x = self.act(x)

		return x

class Respath2_MemOpt(Module):
	def __init__(self, num_in_filters, num_out_filters, respath_length):
	
		super().__init__()
		self.act = torch.nn.SELU(inplace=True)
		self.shortcut1 = Conv2d_batchnorm(num_in_filters, num_out_filters, kernel_size = (1,1), activation=False)
		self.shortcut2 = Conv2d_batchnorm(num_out_filters, num_out_filters, kernel_size = (1,1), activation=False)
		self.conv1 = Conv2d_batchnorm(num_in_filters, num_out_filters, kernel_size = (3,3),activation=True)
		self.conv2 = Conv2d_batchnorm(num_out_filters, num_out_filters, kernel_size = (3,3), activation=True)
		
	def forward(self,x):
		shortcut = self.shortcut1(x)
		x = self.conv1(x)
		x = x + shortcut
		x = self.act(x)

		shortcut = self.shortcut2(x)
		x = self.conv2(x)
		x = x + shortcut
		x = self.act(x)

		return x

class Respath1_MemOpt(Module):
	def __init__(self, num_in_filters, num_out_filters, respath_length):
	
		super().__init__()
		self.act = torch.nn.SELU(inplace=True)
		self.shortcut1 = Conv2d_batchnorm(num_in_filters, num_out_filters, kernel_size = (1,1), activation=False)
		self.conv1 = Conv2d_batchnorm(num_in_filters, num_out_filters, kernel_size = (3,3),activation=True)
		
	def forward(self,x):
		shortcut = self.shortcut1(x)
		x = self.conv1(x)
		x = x + shortcut
		x = self.act(x)

		return x

class MultiResUnet(Module):
	'''
	MultiResUNet
	
	Arguments:
		input_channels {int} -- number of channels in image
		num_classes {int} -- number of segmentation classes
		alpha {float} -- alpha hyperparameter (default: 1.69)
	
	Returns:
		[keras model] -- MultiResUNet model
	'''
	def __init__(self, input_channels, num_classes, alpha=1.69):
		super().__init__()
		
		self.alpha = alpha
		
		# Encoder Path
		self.multiresblock1 = Multiresblock(input_channels,32)
		self.in_filters1 = int(32*self.alpha*0.167)+int(32*self.alpha*0.333)+int(32*self.alpha* 0.5)
		self.pool1 =  torch.nn.MaxPool2d(2)
		self.respath1 = Respath4(self.in_filters1,32,respath_length=4)

		self.multiresblock2 = Multiresblock(self.in_filters1,32*2)
		self.in_filters2 = int(32*2*self.alpha*0.167)+int(32*2*self.alpha*0.333)+int(32*2*self.alpha* 0.5)
		self.pool2 =  torch.nn.MaxPool2d(2)
		self.respath2 = Respath3(self.in_filters2,32*2,respath_length=3)
	
	
		self.multiresblock3 =  Multiresblock(self.in_filters2,32*4)
		self.in_filters3 = int(32*4*self.alpha*0.167)+int(32*4*self.alpha*0.333)+int(32*4*self.alpha* 0.5)
		self.pool3 =  torch.nn.MaxPool2d(2)
		self.respath3 = Respath2(self.in_filters3,32*4,respath_length=2)
	
	
		self.multiresblock4 = Multiresblock(self.in_filters3,32*8)
		self.in_filters4 = int(32*8*self.alpha*0.167)+int(32*8*self.alpha*0.333)+int(32*8*self.alpha* 0.5)
		self.pool4 =  torch.nn.MaxPool2d(2)
		self.respath4 = Respath1(self.in_filters4,32*8,respath_length=1)
	
	
		self.multiresblock5 = Multiresblock(self.in_filters4,32*16)
		self.in_filters5 = int(32*16*self.alpha*0.167)+int(32*16*self.alpha*0.333)+int(32*16*self.alpha* 0.5)
	 
		# Decoder path
		self.upsample6 = torch.nn.ConvTranspose2d(self.in_filters5,32*8,kernel_size=(2,2),stride=(2,2))  
		self.concat_filters1 = 32*8 *2
		self.multiresblock6 = Multiresblock(self.concat_filters1,32*8)
		self.in_filters6 = int(32*8*self.alpha*0.167)+int(32*8*self.alpha*0.333)+int(32*8*self.alpha* 0.5)

		self.upsample7 = torch.nn.ConvTranspose2d(self.in_filters6,32*4,kernel_size=(2,2),stride=(2,2))  
		self.concat_filters2 = 32*4 *2
		self.multiresblock7 = Multiresblock(self.concat_filters2,32*4)
		self.in_filters7 = int(32*4*self.alpha*0.167)+int(32*4*self.alpha*0.333)+int(32*4*self.alpha* 0.5)
	
		self.upsample8 = torch.nn.ConvTranspose2d(self.in_filters7,32*2,kernel_size=(2,2),stride=(2,2))
		self.concat_filters3 = 32*2 *2
		self.multiresblock8 = Multiresblock(self.concat_filters3,32*2)
		self.in_filters8 = int(32*2*self.alpha*0.167)+int(32*2*self.alpha*0.333)+int(32*2*self.alpha* 0.5)
	
		self.upsample9 = torch.nn.ConvTranspose2d(self.in_filters8,32,kernel_size=(2,2),stride=(2,2))
		self.concat_filters4 = 32 *2
		self.multiresblock9 = Multiresblock(self.concat_filters4,32)
		self.in_filters9 = int(32*self.alpha*0.167)+int(32*self.alpha*0.333)+int(32*self.alpha* 0.5)

		self.conv_final = Conv2d_batchnorm(self.in_filters9, num_classes, kernel_size = (1,1), activation=False)

	def forward(self, x):
		x_multires1 = self.multiresblock1(x)
		x_pool1 = self.pool1(x_multires1)
		x_multires1 = self.respath1(x_multires1)
		
		x_multires2 = self.multiresblock2(x_pool1)
		x_pool2 = self.pool2(x_multires2)
		x_multires2 = self.respath2(x_multires2)

		x_multires3 = self.multiresblock3(x_pool2)
		x_pool3 = self.pool3(x_multires3)
		x_multires3 = self.respath3(x_multires3)

		x_multires4 = self.multiresblock4(x_pool3)
		x_pool4 = self.pool4(x_multires4)
		x_multires4 = self.respath4(x_multires4)

		x_multires5 = self.multiresblock5(x_pool4)

		up6 = torch.cat([self.upsample6(x_multires5),x_multires4],axis=1)
		x_multires6 = self.multiresblock6(up6)

		up7 = torch.cat([self.upsample7(x_multires6),x_multires3],axis=1)
		x_multires7 = self.multiresblock7(up7)

		up8 = torch.cat([self.upsample8(x_multires7),x_multires2],axis=1)
		x_multires8 = self.multiresblock8(up8)

		up9 = torch.cat([self.upsample9(x_multires8),x_multires1],axis=1)
		x_multires9 = self.multiresblock9(up9)

		out =  self.conv_final(x_multires9)
		
		return out

class MultiResUnet_MemOpt(Module):
	def __init__(self, input_channels, num_classes, alpha=1.69, msg_queue = None):
		super().__init__()
		
		self.alpha = alpha
		# Encoder Path
		self.multiresblock1 = Multiresblock_MemOpt(input_channels,32)
		self.in_filters1 = int(32*self.alpha*0.167)+int(32*self.alpha*0.333)+int(32*self.alpha* 0.5)
		self.pool1 =  torch.nn.MaxPool2d(2)
		self.respath1 = Respath4_MemOpt(self.in_filters1,32,respath_length=4)

		self.multiresblock2 = Multiresblock_MemOpt(self.in_filters1,32*2)
		self.in_filters2 = int(32*2*self.alpha*0.167)+int(32*2*self.alpha*0.333)+int(32*2*self.alpha* 0.5)
		self.pool2 =  torch.nn.MaxPool2d(2)
		self.respath2 = Respath3_MemOpt(self.in_filters2,32*2,respath_length=3)
	
	
		self.multiresblock3 =  Multiresblock_MemOpt(self.in_filters2,32*4)
		self.in_filters3 = int(32*4*self.alpha*0.167)+int(32*4*self.alpha*0.333)+int(32*4*self.alpha* 0.5)
		self.pool3 =  torch.nn.MaxPool2d(2)
		self.respath3 = Respath2_MemOpt(self.in_filters3,32*4,respath_length=2)
	
	
		self.multiresblock4 = Multiresblock_MemOpt(self.in_filters3,32*8)
		self.in_filters4 = int(32*8*self.alpha*0.167)+int(32*8*self.alpha*0.333)+int(32*8*self.alpha* 0.5)
		self.pool4 =  torch.nn.MaxPool2d(2)
		self.respath4 = Respath1_MemOpt(self.in_filters4,32*8,respath_length=1)
	
		self.multiresblock5 = Multiresblock_MemOpt(self.in_filters4,32*16)
		self.in_filters5 = int(32*16*self.alpha*0.167)+int(32*16*self.alpha*0.333)+int(32*16*self.alpha* 0.5)
		# Decoder path
		self.upsample6 = torch.nn.ConvTranspose2d(self.in_filters5,32*8,kernel_size=(2,2),stride=(2,2))  
		self.concat_filters1 = 32*8 *2
		self.multiresblock6 = Multiresblock_MemOpt(self.concat_filters1,32*8)
		self.in_filters6 = int(32*8*self.alpha*0.167)+int(32*8*self.alpha*0.333)+int(32*8*self.alpha* 0.5)

		self.upsample7 = torch.nn.ConvTranspose2d(self.in_filters6,32*4,kernel_size=(2,2),stride=(2,2))  
		self.concat_filters2 = 32*4 *2
		self.multiresblock7 = Multiresblock_MemOpt(self.concat_filters2,32*4)
		self.in_filters7 = int(32*4*self.alpha*0.167)+int(32*4*self.alpha*0.333)+int(32*4*self.alpha* 0.5)
	
		self.upsample8 = torch.nn.ConvTranspose2d(self.in_filters7,32*2,kernel_size=(2,2),stride=(2,2))
		self.concat_filters3 = 32*2 *2
		self.multiresblock8 = Multiresblock_MemOpt(self.concat_filters3,32*2)
		self.in_filters8 = int(32*2*self.alpha*0.167)+int(32*2*self.alpha*0.333)+int(32*2*self.alpha* 0.5)
	
		self.upsample9 = torch.nn.ConvTranspose2d(self.in_filters8,32,kernel_size=(2,2),stride=(2,2))
		self.concat_filters4 = 32 *2
		self.multiresblock9 = Multiresblock_MemOpt(self.concat_filters4,32)
		self.in_filters9 = int(32*self.alpha*0.167)+int(32*self.alpha*0.333)+int(32*self.alpha* 0.5)

		self.conv_final = Conv2d_batchnorm(self.in_filters9, num_classes, kernel_size = (1,1), activation=False)

		self.msg = Message(msg_queue)

	def forward(self, x):
		try:
			x_multires1 = self.multiresblock1(x)
			x_pool1 = self.pool1(x_multires1)
			x_multires1 = self.respath1(x_multires1)
			
			x_multires2 = self.multiresblock2(x_pool1)
			del x_pool1
			x_pool2 = self.pool2(x_multires2)
			x_multires2 = self.respath2(x_multires2)

			x_multires3 = self.multiresblock3(x_pool2)
			del x_pool2
			x_pool3 = self.pool3(x_multires3)
			x_multires3 = self.respath3(x_multires3)

			x_multires4 = self.multiresblock4(x_pool3)
			del x_pool3
			x_pool4 = self.pool4(x_multires4)
			x_multires4 = self.respath4(x_multires4)

			x_multires5 = self.multiresblock5(x_pool4)
			del x_pool4
			
			up6 = torch.cat([self.upsample6(x_multires5),x_multires4],axis=1)
			x_multires6 = self.multiresblock6(up6)
			del x_multires5
			del x_multires4
			del up6

			up7 = torch.cat([self.upsample7(x_multires6),x_multires3],axis=1)
			x_multires7 = self.multiresblock7(up7)
			del x_multires6
			del x_multires3
			del up7

			up8 = torch.cat([self.upsample8(x_multires7),x_multires2],axis=1)
			x_multires8 = self.multiresblock8(up8)
			del x_multires7
			del x_multires2
			del up8

			up9 = torch.cat([self.upsample9(x_multires8),x_multires1],axis=1)
			x_multires9 = self.multiresblock9(up9)
			del x_multires8
			del x_multires1
			del up9
			out =  self.conv_final(x_multires9)
			return out

		except:
			print (f'GPU Mem low failure')
			x_device = x.device
			x_dtype = x.dtype
			print ('encoder step1 of 5')
			try:
				x_multires1 = self.multiresblock1(x)
			except:
				multiresblock1cpu = self.multiresblock1.to(device='cpu', dtype=torch.float32)
				x_multires1 = multiresblock1cpu(x.to(device='cpu', dtype=torch.float32))
				del multiresblock1cpu
			del x
			try:
				x_pool1 = self.pool1(x_multires1.to(device=x_device, dtype=x_dtype))
			except:
				x_pool1cpu = self.pool1.to(device='cpu', dtype=torch.float32)
				x_pool1 = x_pool1cpu(x_multires1.to(device='cpu', dtype=torch.float32))
				del x_pool1cpu
			try:
				x_multires1 = self.respath1(x_multires1.to(device=x_device, dtype=x_dtype))
			except:
				respath1cpu = self.respath1.to(device='cpu', dtype=torch.float32)
				x_multires1 = respath1cpu(x_multires1.to(device='cpu', dtype=torch.float32))
				del respath1cpu
			x_multires1 = x_multires1.to(device='cpu', dtype=torch.float32)

			print ('encoder step2 of 5')
			try:
				x_multires2 = self.multiresblock2(x_pool1.to(device=x_device, dtype=x_dtype))
			except:
				multiresblock2cpu = self.multiresblock2.to(device='cpu', dtype=torch.float32)
				x_multires2 = multiresblock2cpu(x_pool1.to(device='cpu', dtype=torch.float32))
				del multiresblock2cpu
			del x_pool1
			try:
				x_pool2 = self.pool2(x_multires2.to(device=x_device, dtype=x_dtype))
			except:
				x_pool2cpu = self.pool2.to(device='cpu', dtype=torch.float32)
				x_pool2 = x_pool2cpu(x_multires2.to(device='cpu', dtype=torch.float32))
				del x_pool2cpu
			try:
				x_multires2 = self.respath2(x_multires2.to(device=x_device, dtype=x_dtype))
			except:
				respath2cpu = self.respath2.to(device='cpu', dtype=torch.float32)
				x_multires2 = respath2cpu(x_multires2.to(device='cpu', dtype=torch.float32))
				del respath2cpu
			x_multires2 = x_multires2.to(device='cpu', dtype=torch.float32)

			print ('encoder step3 of 5')
			try:
				x_multires3 = self.multiresblock3(x_pool2.to(device=x_device, dtype=x_dtype))
			except:
				multiresblock3cpu = self.multiresblock3.to(device='cpu', dtype=torch.float32)
				x_multires3 = multiresblock3cpu(x_pool2.to(device='cpu', dtype=torch.float32))
				del multiresblock3cpu
			del x_pool2
			try:
				x_pool3 = self.pool3(x_multires3.to(device=x_device, dtype=x_dtype))
			except:
				x_pool3cpu = self.pool3.to(device='cpu', dtype=torch.float32)
				x_pool3 = x_pool3cpu(x_multires3.to(device='cpu', dtype=torch.float32))
				del x_pool3cpu
			try:
				x_multires3 = self.respath3(x_multires3.to(device=x_device, dtype=x_dtype))
			except:
				respath3cpu = self.respath3.to(device='cpu', dtype=torch.float32)
				x_multires3 = respath3cpu(x_multires3.to(device='cpu', dtype=torch.float32))
				del respath3cpu
			x_multires3 = x_multires3.to(device='cpu', dtype=torch.float32)

			print ('encoder step4 of 5')
			try:
				x_multires4 = self.multiresblock4(x_pool3.to(device=x_device, dtype=x_dtype))
			except:
				multiresblock4cpu = self.multiresblock4.to(device='cpu', dtype=torch.float32)
				x_multires4 = multiresblock4cpu(x_pool3.to(device='cpu', dtype=torch.float32))
				del multiresblock4cpu
			del x_pool3
			try:
				x_pool4 = self.pool4(x_multires4.to(device=x_device, dtype=x_dtype))
			except:
				x_pool4cpu = self.pool4.to(device='cpu', dtype=torch.float32)
				x_pool4 = x_pool4cpu(x_multires4.to(device='cpu', dtype=torch.float32))
				del x_pool4cpu
			try:
				x_multires4 = self.respath4(x_multires4.to(device=x_device, dtype=x_dtype))
			except:
				respath4cpu = self.respath4.to(device='cpu', dtype=torch.float32)
				x_multires4 = respath4cpu(x_multires4.to(device='cpu', dtype=torch.float32))
				del respath4cpu
			x_multires4 = x_multires4.to(device='cpu', dtype=torch.float32)

			print ('encoder step5 of 5')
			try:
				x_multires5 = self.multiresblock5(x_pool4.to(device=x_device, dtype=x_dtype))
			except:
				multiresblock5cpu = self.multiresblock5.to(device='cpu', dtype=torch.float32)
				x_multires5 = multiresblock5cpu(x_pool4.to(device='cpu', dtype=torch.float32))
			x_multires5 = x_multires5.to(device='cpu', dtype=torch.float32)
			del x_pool4

			print ('encoder completed')

			print ('decoder step 1 of 5')
			try:
				up6 = torch.cat([self.upsample6(x_multires5.to(device=x_device, dtype=x_dtype)),x_multires4.to(device=x_device, dtype=x_dtype)],axis=1)
			except:
				upsample6cpu = self.upsample6.to(device='cpu', dtype=torch.float32)
				up6 = torch.cat([upsample6cpu(x_multires5.to(device='cpu', dtype=torch.float32)),x_multires4.to(device='cpu', dtype=torch.float32)],axis=1)
				del upsample6cpu
			try:
				x_multires6 = self.multiresblock6(up6.to(device=x_device, dtype=x_dtype))
			except:
				multiresblock6cpu = self.multiresblock6.to(device='cpu', dtype=torch.float32)
				x_multires6 = multiresblock6cpu(up6.to(device='cpu', dtype=torch.float32))
			del x_multires5, x_multires4, up6

			print ('decoder step 2 of 5')
			try:
				up7 = torch.cat([self.upsample7(x_multires6.to(device=x_device, dtype=x_dtype)),x_multires3.to(device=x_device, dtype=x_dtype)],axis=1)
			except:
				upsample7cpu = self.upsample7.to(device='cpu', dtype=torch.float32)
				up7 = torch.cat([upsample7cpu(x_multires6.to(device='cpu', dtype=torch.float32)),x_multires3.to(device='cpu', dtype=torch.float32)],axis=1)
				del upsample7cpu
			try:
				x_multires7 = self.multiresblock7(up7.to(device=x_device, dtype=x_dtype))
			except:
				multiresblock7cpu = self.multiresblock7.to(device='cpu', dtype=torch.float32)
				x_multires7 = multiresblock7cpu(up7.to(device='cpu', dtype=torch.float32))
			del x_multires6, x_multires3, up7

			print ('decoder step 3 of 5')
			try:
				up8 = torch.cat([self.upsample8(x_multires7.to(device=x_device, dtype=x_dtype)),x_multires2.to(device=x_device, dtype=x_dtype)],axis=1)
			except:
				upsample8cpu = self.upsample8.to(device='cpu', dtype=torch.float32)
				up8 = torch.cat([upsample8cpu(x_multires7.to(device='cpu', dtype=torch.float32)),x_multires2.to(device='cpu', dtype=torch.float32)],axis=1)
				del upsample8cpu
			try:
				x_multires8 = self.multiresblock8(up8.to(device=x_device, dtype=x_dtype))
			except:
				multiresblock8cpu = self.multiresblock8.to(device='cpu', dtype=torch.float32)
				x_multires8 = multiresblock8cpu(up8.to(device='cpu', dtype=torch.float32))
			del x_multires7, x_multires2, up8

			print ('decoder step 4 of 5')
			try:
				up9 = torch.cat([self.upsample9(x_multires8.to(device=x_device, dtype=x_dtype)),x_multires1.to(device=x_device, dtype=x_dtype)],axis=1)
			except:
				upsample9cpu = self.upsample9.to(device='cpu', dtype=torch.float32)
				up9 = torch.cat([upsample9cpu(x_multires8.to(device='cpu', dtype=torch.float32)),x_multires1.to(device='cpu', dtype=torch.float32)],axis=1)
				del upsample9cpu
			try:
				x_multires9 = self.multiresblock9(up9.to(device=x_device, dtype=x_dtype))
			except:
				multiresblock9cpu = self.multiresblock9.to(device='cpu', dtype=torch.float32)
				x_multires9 = multiresblock9cpu(up9.to(device='cpu', dtype=torch.float32))
			del x_multires8, x_multires1, up9

			print ('decoder step 5 of 5')
			try:
				out =  self.conv_final(x_multires9.to(device=x_device, dtype=x_dtype))
			except:
				conv_final_cpu = self.conv_final.to(device='cpu', dtype=torch.float32)
				out = conv_final_cpu(x_multires9.to(device='cpu', dtype=torch.float32))

			print ('decoder completed')

			return out.to(device=x_device, dtype=x_dtype)

class Message:
	def __init__(self, queue) -> None:
		self.queue = queue
	def send(msg):
		pass

class Model:
    @staticmethod
    def get_name():
        return 'MultiRes_v001'

    @staticmethod
    def get_model():
        return MultiResUnet_MemOpt
	
    @staticmethod
    def get_training_model():
        return MultiResUnet
