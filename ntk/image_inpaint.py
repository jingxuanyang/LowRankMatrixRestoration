import utilities.load_image as loader
import mpld3
# mpld3.enable_notebook()

fname = 'data/car.jpg'
clean, corrupted, mask = loader.load_image(fname, img_width=128, img_height=128)
loader.visualize_corrupted_image(clean, corrupted)

import construct_kernel.layers as nn

# Construct a network for which we will compute the CNTK 
net = nn.Sequential(nn.Conv(kernel_size=3),
                    nn.LeakyReLU(),
                    nn.Downsample(),
                    nn.Conv(kernel_size=3),
                    nn.LeakyReLU(),                    
                    nn.Downsample(),
                    nn.Conv(kernel_size=3),
                    nn.LeakyReLU(),
                    nn.Downsample(),      
                    nn.Conv(kernel_size=3),
                    nn.LeakyReLU(),
                    nn.Upsample(bilinear=False),                    
                    nn.Conv(kernel_size=3),
                    nn.LeakyReLU(),
                    nn.Upsample(bilinear=False),
                    nn.Conv(kernel_size=3),
                    nn.LeakyReLU(),
                    nn.Upsample(bilinear=False),
                    nn.Conv(kernel_size=3))

# Provide an image size, number of CPU threads, and possibly a feature prior X for computing the kernel 
image_size = 16
K = net.get_ntk(image_size, num_threads=10)

print(K.shape)

import kr_solvers.expander as expander
import kr_solvers.exact_solve as exact_solve
import utilities.visualizer as vis

s = 3  # Number of downsampling / upsampling layers in network
_, w, h = corrupted.shape  # Shape to expand our kernel to (we only need width and height)
K_expanded = expander.expand_kernel(K, s, w, h)
imputed_img_exact = exact_solve.kr_solve(K_expanded, corrupted, mask)
vis.visualize_images(corrupted, imputed_img_exact)

import utilities.visualizer as vis

vis.visualize_kernel_slice(K_expanded, (16, 16))

import kr_solvers.eigenpro_solve as eigenpro_solve
import utilities.visualizer as vis

s = 3  # Number of downsampling / upsampling layers in network
imputed_img = eigenpro_solve.expand_and_solve(K, corrupted, mask, s, max_iter=20)
vis.visualize_images(corrupted, imputed_img)

import utilities.load_image as loader
import hickle

fname = 'data/hill_GT.png'
clean, corrupted, mask = loader.load_image(fname, img_width=128, img_height=128)

# 6 down/upsampling layers, meshgrid feature prior
K = hickle.load('saved_kernels/128x128_6ud_dip_meshgrid.h')  

# Exact solver for cached kernel
import kr_solvers.expander as expander
import kr_solvers.exact_solve as exact_solve
import utilities.visualizer as vis

imputed_img_exact = exact_solve.kr_solve(K, corrupted, mask)
vis.visualize_images(corrupted, imputed_img_exact)

# Visualize kernel slices for expanded kernel
vis.visualize_kernel_slice(K, (16, 16))

# EigenPro solve for cached kernel
imputed_img = eigenpro_solve.solve(K, corrupted, mask, max_iter=100)
vis.visualize_images(corrupted, imputed_img)
