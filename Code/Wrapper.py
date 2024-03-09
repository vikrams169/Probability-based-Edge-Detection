#!/usr/bin/env python

#Importing the required libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import imutils
import sklearn.cluster
import cv2

#Paths to the folders containing the images and their respective Canny and Sobel Baselines
image_folder = "../BSDS500/Images"
canny_baseline_folder = "../BSDS500/CannyBaseline"
sobel_baseline_folder = "../BSDS500/SobelBaseline"

#Parameters being used for various operations to generate the pblite images
filter_size = [21]
sigma_dog = [2,3]
sigma_lms = [1,2**0.5,2,2*(2**0.5)]
sigma_lml = [2**0.5,2,2*(2**0.5),4]
sigma_gabor = [10,25]
orientations_6 = [0, 60, 120, 180, 240, 300]
orientations_16 = [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5, 180, 202.5, 225, 247.5, 270, 292.5, 315, 337.5]
frequencies_gabor = [2,3,4]
radii_half_disks = [2, 5, 10, 20, 30]
num_clusters_texton = 64
num_clusters_brightness = 16
num_clusters_color = 16
canny_sobel_weights = [0.5, 0.5]

#Reading the images and their respective Canny and Sobel Baselines as lists
def images_from_folders(image_folder,canny_folder,sobel_folder):
      images = []
      canny_images = []
      sobel_images = []
      filenames = os.listdir(image_folder)
      for filename in filenames:
            edge_filename = filename[:-3] + "png"
            image = cv2.imread(image_folder + '/' + filename)
            canny_image = cv2.imread(canny_folder + "/" + edge_filename)
            sobel_image = cv2.imread(sobel_folder + "/" + edge_filename)
            images.append(image)
            canny_images.append(canny_image)
            sobel_images.append(sobel_image)
      return images, canny_images, sobel_images
        

#Displaying (and optionally saving) the filter banks
def display_filters(filters,title,num_rows,destination=None):
      num_cols = len(filters)//num_rows
      plt.subplots(num_rows,num_cols,figsize=(15,15))
      plt.title(title)
      for i in range(len(filters)):
            plt.subplot(num_rows,num_cols,i+1)
            plt.axis('off')
            plt.imshow(filters[i],cmap='gray')
      if destination is not None:
            plt.savefig(destination)
      plt.show()

#Saving an image to a particular folder/path
def save_images(images,folder,grayscale=0):
      for i in range(len(images)):
            if grayscale==0:
                  plt.imshow(images[i])
                  plt.savefig(folder+str(i+1)+'.png')
            else:
                  plt.imshow(images[i],cmap='gray')
                  plt.savefig(folder+str(i+1)+'.png')
    
#A Sobel Filter in the horizontal/X direction
def sobel_filter_horizontal():
      return np.array([[-1,0,1],[-2,0,2],[-1,0,1]])

#A Sobel Filter in the vertical/Y direction
def sobel_filter_vertical():
      return np.array([[-1,-2,-1],[0,0,0],[1,2,1]])

#A Laplacian of Gaussian (LoG) Filter
def laplacian_of_gaussian_filter():
      return np.array([[0,1,0],[1,-4,1],[0,1,0]])

#Generating a 2D sinusoidal filter
def sin_2d_filter(filter_size, theta, frequency):
      center = filter_size//2
      filter_x = np.zeros((filter_size,filter_size))
      filter_y = np.zeros((filter_size,filter_size))
      for i in range(filter_size):
            for j in range(filter_size):
                  filter_x[i,j] = i-center
                  filter_y[i,j] = j-center
      filter = np.cos(np.deg2rad(theta))*filter_x + np.sin(np.deg2rad(theta))*filter_y
      return np.sin(2*(np.pi)*(frequency/filter_size)*filter)                 

#Generating a 2D Gaussian Filter
def gaussian_filter(filter_size, sigma_x, sigma_y):
      center = filter_size//2
      filter = np.zeros((filter_size,filter_size))
      for i in range(filter_size):
            for j in range(filter_size):
                  filter[i,j] = (1/(2*np.pi*(sigma_x*sigma_y)))*np.exp(-((((i-center)**2)/(2*sigma_x))+(((j-center)**2))/(2*sigma_y)))
      return filter

#Custom 2D convolution implementation (not as efficient as cv2.filter2D())
def convolve_2d(image, filter):
      padding = (filter.shape[0]-1)//2
      new_image = np.zeros((image.shape[0],image.shape[1]))
      padded_image = np.pad(image,((padding,padding),(padding,padding)),'constant')
      for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                  new_image[i,j] = np.sum(padded_image[i:i+filter.shape[0],j:j+filter.shape[1]]*filter)
      return new_image        

#Generating a single derivative of Gaussian (DoG) Filter
def single_DoG(filter_size,sigma,theta):
      filter_x = cv2.filter2D(gaussian_filter(filter_size,sigma,sigma),-1,sobel_filter_horizontal())
      filter_y = cv2.filter2D(gaussian_filter(filter_size,sigma,sigma),-1,sobel_filter_vertical())
      return np.cos(np.deg2rad(theta))*filter_x + np.sin(np.deg2rad(theta))*filter_y
      
#Generating a set of DoG Filters
def multiple_DoG(filter_sizes, sigmas, thetas):
      filters = []
      for i in range(len(filter_sizes)):
            for j in range(len(sigmas)):
                  for k in range(len(thetas)):
                        filters.append(single_DoG(filter_sizes[i],sigmas[j],thetas[k]))
      return filters

#Generating the first and second derivative filters (a part of LM Filters)
def LM_derivatives(filter_sizes, sigmas, thetas):
      filters = []
      for i in range(len(filter_sizes)):
            for j in range(len(sigmas)):
                  gaussian_filter_base = gaussian_filter(filter_sizes[i],3*sigmas[j],sigmas[j])
                  filter_d1_x = convolve_2d(gaussian_filter_base,sobel_filter_horizontal())
                  filter_d1_y = convolve_2d(gaussian_filter_base,sobel_filter_vertical())
                  filter_d1 = filter_d1_x + filter_d1_y
                  filter_d2_x = convolve_2d(filter_d1,sobel_filter_horizontal())
                  filter_d2_y = convolve_2d(filter_d1,sobel_filter_vertical())
                  filter_d2 = filter_d2_x + filter_d2_y
                  for k in range(len(thetas)):
                        filter_d1_theta = np.cos(np.deg2rad(thetas[k]))*filter_d1_x + np.sin(np.deg2rad(thetas[k]))*filter_d1_y
                        filter_d2_theta = np.cos(np.deg2rad(thetas[k]))*filter_d2_x + np.sin(np.deg2rad(thetas[k]))*filter_d2_y
                        filters.append(filter_d1_theta)
                        filters.append(filter_d2_theta)
      return filters

#Generating the LoG Filters (a part of LM Filters)
def LM_LoG(filter_sizes, sigmas):
      filters = []
      for i in range(len(filter_sizes)):
            for j in range(len(sigmas)):
                  filters.append(cv2.filter2D(gaussian_filter(filter_sizes[i],sigmas[j],sigmas[j]),-1,laplacian_of_gaussian_filter()))
                  filters.append(cv2.filter2D(gaussian_filter(filter_sizes[i],3*sigmas[j],3*sigmas[j]),-1,laplacian_of_gaussian_filter()))
      return filters

#Generating the Gaussian Filters (a part of LM Filters)
def LM_gaussian(filter_sizes, sigmas):
      filters = []
      for i in range(len(filter_sizes)):
            for j in range(len(sigmas)):
                  filters.append(gaussian_filter(filter_sizes[i],sigmas[j],sigmas[j]))
      return filters

#Generating the set of Leung-Malik (LM) Filters
def LM_filters(filter_sizes,sigmas,thetas):
      filters = []
      filters.extend(LM_derivatives(filter_sizes,sigmas[0:3],thetas))
      filters.extend(LM_LoG(filter_sizes,sigmas))
      filters.extend(LM_gaussian(filter_sizes,sigmas))
      return filters

#Generating the set of Gabor Filters
def gabor_filters(filter_sizes,sigmas,thetas,frequencies):
      filters = []
      for i in range(len(filter_sizes)):
            for j in range(len(sigmas)):
                  gaussian_filter_base = gaussian_filter(filter_sizes[i],sigmas[j],sigmas[j])
                  for k in range(len(thetas)):
                        for l in range(len(frequencies)):
                              sin_filter = sin_2d_filter(filter_sizes[i],thetas[k],frequencies[l])
                              filters.append(gaussian_filter_base*sin_filter)
      return filters

#Generating a single half-disk filter
def single_half_disk(radius, theta):
      filter_size = (2*radius)+1
      filter = np.zeros((filter_size,filter_size))
      for i in range(filter_size):
            for j in range(filter_size):
                  if np.sqrt(((i-radius)**2)+((j-radius)**2)) <= radius and i<=radius:
                        filter[i,j] = 1
      filter = imutils.rotate(filter,theta)
      filter[filter>0.5] = 1
      filter[filter<=0.5] = 0
      return filter

#Generating the set of half-disk filters
def half_disk_filters(radii, thetas):
      filters = []
      for i in range(len(radii)):
            unordered_filters = []
            for j in range(len(thetas)):
                  filter = single_half_disk(radii[i],thetas[j])
                  unordered_filters.append(filter)
            for j in range(len(thetas)//2):
                  filters.append(unordered_filters[j])
                  filters.append(unordered_filters[j+((len(thetas))//2)])
      return filters
                                  
#Applying/Convolving all the filters in the filter bank over an image
def apply_all_filters(image,filter_bank):
      filtered_image_stack = []
      image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
      for i in range(len(filter_bank)):
            filtered_image_stack.append(cv2.filter2D(image_gray,-1,filter_bank[i]))
      return filtered_image_stack

#Generating the Texton Map
def texton_map(filtered_image_stack,num_clusters):
      filtered_image_stack = np.array(filtered_image_stack)
      num_filters, height, width = filtered_image_stack.shape
      filtered_image_stack = np.transpose(filtered_image_stack.reshape(num_filters,height*width))
      clustering_model = sklearn.cluster.KMeans(n_clusters=num_clusters,n_init=2)
      clustering_model.fit(filtered_image_stack)
      pixel_labels = clustering_model.predict(filtered_image_stack)
      texton_map = pixel_labels.reshape([height,width])
      return texton_map

#Generating the Brightness Map
def brightness_map(image,num_clusters):
      image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
      height, width = image_gray.shape
      image_gray = image_gray.reshape([height*width,1])
      clustering_model = sklearn.cluster.KMeans(n_clusters=num_clusters,n_init=2)
      clustering_model.fit(image_gray)
      pixel_labels = clustering_model.predict(image_gray)
      brightness_map = pixel_labels.reshape([height,width])
      return brightness_map

#Generating the Color Map
def color_map(image,num_clusters):
      height, width, channels = image.shape
      image_resized = image.reshape([height*width,channels])
      clustering_model = sklearn.cluster.KMeans(n_clusters=num_clusters,n_init=2)
      clustering_model.fit(image_resized)
      pixel_labels = clustering_model.predict(image_resized)
      color_map = pixel_labels.reshape([height,width])
      return color_map

#Computing the Map Gradients (for a texton/brightness/color map)
def map_to_gradients(map,half_disk_filters,num_clusters):
      chi_square_distances = []
      minimum_assigned_cluster = np.min(map)
      index = 0
      while index < len(half_disk_filters):
            filter_left = half_disk_filters[index]
            filter_right = half_disk_filters[index+1]
            filter_chi_square_distances = np.zeros(((map.shape[0],map.shape[1])))
            for j in range(num_clusters):
                  temp_map = np.zeros((map.shape[0],map.shape[1]))
                  temp_map[map == (j+minimum_assigned_cluster)] = 1
                  img_g = cv2.filter2D(temp_map,-1,filter_left)
                  img_h = cv2.filter2D(temp_map,-1,filter_right)
                  filter_chi_square_distances += 0.5*((img_g-img_h)**2)/(img_g+img_h+np.exp(-5))
            chi_square_distances.append(filter_chi_square_distances)
            index += 2
      chi_square_distances = np.mean(np.array(chi_square_distances),axis=0)
      return chi_square_distances

#Generating the pblite images from the map gradients and Canny/Sobel baselines
def detect_edges(map_gradients,canny_edges,sobel_edges,weights):
      #texton_gradients = map_gradients[0]
      #brightness_gradients = map_gradients[1]
      #color_gradients = map_gradients[2]
      #canny_weight = weights[0]
      #sobel_weight = weights[1]
      canny_edges = cv2.cvtColor(canny_edges,cv2.COLOR_BGR2GRAY)
      sobel_edges = cv2.cvtColor(sobel_edges,cv2.COLOR_BGR2GRAY)
      image_edges = (1/3)*(map_gradients[0]+map_gradients[1]+map_gradients[2])*((weights[0]*canny_edges)+(weights[1]*sobel_edges))
      return image_edges

def main():
      
      # Initializing the Filter Bank
      filter_bank = []

      # Generating the Derivative of Gaussian (DoG) Filters
      DoG_filter_set = multiple_DoG(filter_size,sigma_dog,orientations_16)
      filter_bank.extend(DoG_filter_set)
      display_filters(DoG_filter_set,'Derivative of Gaussian (DoG) Filter Set',4,'results/filters/DoG.png')
      print("Generated the DoG Filters")
      
      # Generating the Leung-Malik (LM) Filters
      # Contain both Leung-Malik Small (LMS) and Leung-Malik Large (LML) Filters
      LM_filter_set = []
      LMS_filter_set = LM_filters(filter_size,sigma_lms,orientations_6)
      display_filters(LMS_filter_set,'Leung-Malik Small (LMS) Filter Set',6,'results/filters/LMS.png')
      LML_filter_set = LM_filters(filter_size,sigma_lml,orientations_6)
      display_filters(LML_filter_set,'Leung-Malik Large (LML) Filter Set',6,'results/filters/LML.png')
      LM_filter_set.extend(LMS_filter_set)
      LM_filter_set.extend(LML_filter_set)
      filter_bank.extend(LM_filter_set)
      display_filters(LM_filter_set,'Overall Leung-Malik (LM) Filter Set',12,'results/filters/LM.png')
      print("Generated the LM Filters")
      
      # Generating the Gabor Filters
      gabor_filter_set = gabor_filters(filter_size,sigma_gabor,orientations_6,frequencies_gabor)
      filter_bank.extend(gabor_filter_set)
      display_filters(gabor_filter_set,'Gabor Filter Set',6,'results/filters/Gabor.png')
      print("Generated the Gabor Filters")
      
      # Generating the Half-Disk Masks
      half_disk_filter_set = half_disk_filters(radii_half_disks,orientations_16)
      display_filters(half_disk_filter_set,'Half-Disk Filter Set',10,'results/filters/HalfDisks.png')
      print("Generated the half-disk filters")

      # Reading the Images and their respective Canny and Sobel Baselines
      images, canny_edges, sobel_edges = images_from_folders(image_folder,canny_baseline_folder,sobel_baseline_folder)
      num_images = len(images)
      filtered_image_stack = []
      for i in range(num_images):
            filtered_image_stack.append(apply_all_filters(images[i],filter_bank))
      print("Read all the images and their respective Canny and Sobel baselines")

      # Generating the Texton Maps
      texton_maps = []
      for i in range(num_images):
            texton_maps.append(texton_map(filtered_image_stack[i],num_clusters_texton))
      save_images(texton_maps,'results/texton_maps/')
      print("Generated the Texton Maps")
      
      # Generating the Texton Gradients
      texton_gradients = []
      for i in range(num_images):
            texton_gradients.append(map_to_gradients(texton_maps[i],half_disk_filter_set,num_clusters_texton))
      save_images(texton_gradients,'results/texton_gradients/')
      print("Computed the Texton Gradients")
      
      # Generating the Brightness Maps
      brightness_maps = []
      for i in range(num_images):
            brightness_maps.append(brightness_map(images[i],num_clusters_brightness))
      save_images(brightness_maps,'results/brightness_maps/')
      print("Generated the Brightness Maps")
      
      # Generating the Brightness Gradients
      brightness_gradients = []
      for i in range(num_images):
            brightness_gradients.append(map_to_gradients(brightness_maps[i],half_disk_filter_set,num_clusters_brightness))
      save_images(brightness_gradients,'results/brightness_gradients/')
      print("Computed the Brightness Gradients")
      
      # Generating the Color Maps
      color_maps = []
      for i in range(num_images):
            color_maps.append(color_map(images[i],num_clusters_color))
      save_images(color_maps,'results/color_maps/')
      print("Generated the Color Maps")
      
      # Generating the Color Gradients
      color_gradients = []
      for i in range(num_images):
            color_gradients.append(map_to_gradients(color_maps[i],half_disk_filter_set,num_clusters_color))
      save_images(color_gradients,'results/color_gradients/')
      print("Computed the Color Gradients")
      
      # Combining the Set of Gradients Together for each Image
      map_gradients = []
      for i in range(num_images):
            map_gradients.append([])
            map_gradients[i].append(texton_gradients[i])
            map_gradients[i].append(brightness_gradients[i])
            map_gradients[i].append(color_gradients[i])
      
      # Generating and Saving the Pb-Lite Outputs
      pblite_images = []
      for i in range(num_images):
            pblite_images.append(detect_edges(map_gradients[i],canny_edges[i],sobel_edges[i],canny_sobel_weights))
      save_images(pblite_images,'results/pblite_images/',1)
      print("Generated and saved the pblite images")
    
if __name__ == '__main__':
    main()

