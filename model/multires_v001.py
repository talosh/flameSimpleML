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
		x_device = x.device
		x_dtype = x.dtype
		try:
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
		except:
			print ('flameSimpleML Multiresblock: Low GPU memory - trying mixed mode (slow)')
			device = x.device
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
				torch.cuda.empty_cache()
			except:
				pass
			try:
				x = x_cpu.to(device=x_device, dtype=x_dtype)
				del x_cpu
				return x
			except:
				return x_cpu

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
	'''
	ResPath
	
	Arguments:
		num_in_filters {int} -- Number of filters going in the respath
		num_out_filters {int} -- Number of filters going out the respath
		respath_length {int} -- length of ResPath
		
	'''

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
	'''
	ResPath
	
	Arguments:
		num_in_filters {int} -- Number of filters going in the respath
		num_out_filters {int} -- Number of filters going out the respath
		respath_length {int} -- length of ResPath
		
	'''

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
		x = self.conv1(x)
		x = x + self.shortcut1(x)
		x = self.act(x)

		x = self.conv2(x)
		x = x + self.shortcut2(x)
		x = self.act(x)

		x = self.conv3(x)
		x = x + self.shortcut3(x)
		x = self.act(x)

		return x

class Respath2(Module):
	'''
	ResPath
	
	Arguments:
		num_in_filters {int} -- Number of filters going in the respath
		num_out_filters {int} -- Number of filters going out the respath
		respath_length {int} -- length of ResPath
		
	'''

	def __init__(self, num_in_filters, num_out_filters, respath_length):
	
		super().__init__()
		self.act = torch.nn.SELU()
		self.shortcut1 = Conv2d_batchnorm(num_in_filters, num_out_filters, kernel_size = (1,1), activation=False)
		self.shortcut2 = Conv2d_batchnorm(num_out_filters, num_out_filters, kernel_size = (1,1), activation=False)
		self.conv1 = Conv2d_batchnorm(num_in_filters, num_out_filters, kernel_size = (3,3),activation=True)
		self.conv2 = Conv2d_batchnorm(num_out_filters, num_out_filters, kernel_size = (3,3), activation=True)
		
	def forward(self,x):
		x = self.conv1(x)
		x = x + self.shortcut1(x)
		x = self.act(x)

		x = self.conv2(x)
		x = x + self.shortcut2(x)
		x = self.act(x)

		return x

class Respath1(Module):
	'''
	ResPath
	
	Arguments:
		num_in_filters {int} -- Number of filters going in the respath
		num_out_filters {int} -- Number of filters going out the respath
		respath_length {int} -- length of ResPath
		
	'''

	def __init__(self, num_in_filters, num_out_filters, respath_length):
	
		super().__init__()
		self.act = torch.nn.SELU()
		self.shortcut1 = Conv2d_batchnorm(num_in_filters, num_out_filters, kernel_size = (1,1), activation=False)
		self.conv1 = Conv2d_batchnorm(num_in_filters, num_out_filters, kernel_size = (3,3),activation=True)
		
	def forward(self,x):
		x = self.conv1(x)
		x = x + self.shortcut1(x)
		x = self.act(x)

		return x

class Respath_MemOpt(Module):
	'''
	ResPath
	
	Arguments:
		num_in_filters {int} -- Number of filters going in the respath
		num_out_filters {int} -- Number of filters going out the respath
		respath_length {int} -- length of ResPath
		
	'''

	def __init__(self, num_in_filters, num_out_filters, respath_length):
		super().__init__()
		print (f'respath memopt init {respath_length}')
		self.respath_length = respath_length
		self.shortcuts = torch.nn.ModuleList([])
		self.convs = torch.nn.ModuleList([])
		self.bns = torch.nn.ModuleList([])
		self.act = torch.nn.SELU(inplace=True)

		for i in range(self.respath_length):
			if(i==0):
				self.shortcuts.append(Conv2d_batchnorm(num_in_filters, num_out_filters, kernel_size = (1,1), activation=False))
				self.convs.append(Conv2d_batchnorm(num_in_filters, num_out_filters, kernel_size = (3,3), activation=True, inplace=True))
			else:
				self.shortcuts.append(Conv2d_batchnorm(num_out_filters, num_out_filters, kernel_size = (1,1), activation=False))
				self.convs.append(Conv2d_batchnorm(num_out_filters, num_out_filters, kernel_size = (3,3), activation=True, inplace=True))
		
	def forward(self, x):
		x_device = x.device
		x_dtype = x.dtype

		print ('respath forward')
		print (f'x shape: {x.shape}')
		print (f'respath length: {self.respath_length}')

		for i in range(self.respath_length):
			print (f'respath iteration: {i}')
			print (f'x shape: {x.shape}')

			try:
				shortcut = self.shortcuts[i](x)
				x = self.convs[i](x)
				# x = self.act(x)
				x = x + shortcut
				del shortcut
				x = self.act(x)
			except:
				print (f'out of mem in respath {i}')
				try:
					shortcut = self.shortcuts[i](x)
					shortcut_cpu = shortcut.cpu()
					del shortcut
				except:
					print (f'out of mem shortcut in respath {i} - cpu')
					shortcut_conv_cpu = self.shortcuts[i].cpu()
					shortcut_conv_cpu = shortcut_conv_cpu.to(torch.float32)
					x_cpu = x
					x_cpu = x_cpu.to(device='cpu', dtype=torch.float32)
					shortcut_cpu = shortcut_conv_cpu(x_cpu)
					del x_cpu
				try:
					x = self.convs[i](x)
					x_cpu = x.cpu()
					del x
				except:
					print (f'out of mem conv in respath {i} - cpu')
					x_cpu = x
					del x
					x_cpu = x_cpu.to(device='cpu', dtype=torch.float32)
					conv_cpu = self.convs[i].cpu()
					conv_cpu = conv_cpu.to(torch.float32)
					x_cpu = conv_cpu(x_cpu)
				x_cpu = x_cpu + shortcut_cpu
				x_cpu = self.act(x_cpu)
				x = x_cpu.to(x_device, dtype=x_dtype)

			try:
				torch.cuda.empty_cache()
			except:
				pass
			
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
		self.respath1 = Respath_MemOpt(self.in_filters1,32,respath_length=4)

		self.multiresblock2 = Multiresblock_MemOpt(self.in_filters1,32*2)
		self.in_filters2 = int(32*2*self.alpha*0.167)+int(32*2*self.alpha*0.333)+int(32*2*self.alpha* 0.5)
		self.pool2 =  torch.nn.MaxPool2d(2)
		self.respath2 = Respath_MemOpt(self.in_filters2,32*2,respath_length=3)
	
	
		self.multiresblock3 =  Multiresblock_MemOpt(self.in_filters2,32*4)
		self.in_filters3 = int(32*4*self.alpha*0.167)+int(32*4*self.alpha*0.333)+int(32*4*self.alpha* 0.5)
		self.pool3 =  torch.nn.MaxPool2d(2)
		self.respath3 = Respath_MemOpt(self.in_filters3,32*4,respath_length=2)
	
	
		self.multiresblock4 = Multiresblock_MemOpt(self.in_filters3,32*8)
		self.in_filters4 = int(32*8*self.alpha*0.167)+int(32*8*self.alpha*0.333)+int(32*8*self.alpha* 0.5)
		self.pool4 =  torch.nn.MaxPool2d(2)
		self.respath4 = Respath_MemOpt(self.in_filters4,32*8,respath_length=1)
	
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
		print ('enc start')
		x_multires1 = self.multiresblock1(x)
		print ('enc multiresblock1')
		x_pool1 = self.pool1(x_multires1)
		print ('enc pool1')
		x_multires1 = self.respath1(x_multires1)

		print ('enc step 01')
		
		x_multires2 = self.multiresblock2(x_pool1)
		del x_pool1
		x_pool2 = self.pool2(x_multires2)
		x_multires2 = self.respath2(x_multires2)

		print ('enc step 02')

		x_multires3 = self.multiresblock3(x_pool2)
		del x_pool2
		x_pool3 = self.pool3(x_multires3)
		x_multires3 = self.respath3(x_multires3)

		print ('enc step 03')

		x_multires4 = self.multiresblock4(x_pool3)
		del x_pool3
		x_pool4 = self.pool4(x_multires4)
		x_multires4 = self.respath4(x_multires4)

		print ('enc step 04')

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
