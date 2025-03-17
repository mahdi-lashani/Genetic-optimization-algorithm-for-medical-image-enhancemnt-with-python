import numpy as np
import cv2
import skimage 
from skimage import filters, io,  color, feature, measure
from skimage.filters import threshold_otsu
import copy
import random
from array import *
import math
################################################################

# initialization 
FCR = 0 # Fitness Computation Rate
FCR_max = 100 # Maximum Fitness Computation Rate
fitness_global_best = -math.inf 
X_best = [0, 0, 0, 0] #[a, b, c, k]
POP = 100 # population size
n = 3 # the size of the neighborhood 
MSE = 0 # Mean Square Error
lb_a = 2   # limitations for our prameters
ub_a = 2.5 # limitations for our prameters
lb_b = 0.3 # limitations for our prameters
ub_b = 0.5 # limitations for our prameters
lb_c = 0   # limitations for our prameters
ub_c = 3   # limitations for our prameters
lb_k = 3   # limitations for our prameters
ub_k = 4   # limitations for our prameters
FI = 0 # the number of foreground pixels φg 

X = np.zeros((100, 4), dtype="float") # structure of X or our random parameters
fitness = np.zeros((100, 1), dtype="float")
sum_fitness = 0

# GENETIC variables and constants
new_X = []       # the better population created by offsprings
new_fitness = [] # fitness of the new population 
elites = np.zeros((1, 50), dtype="float")         # i believe it works even if we assing [] to elites_fitness
elites_fitness = np.zeros((1, 50), dtype="float") # i believe it works even if we assing [] to elites_fitness
L = np.zeros((1, 4), dtype="float")               # L is the smallest parameter
L = [math.inf, math.inf, math.inf, math.inf]      # initializing L
U = np.zeros((1, 4), dtype="float")               # U is the largest parameter
U = [-math.inf, -math.inf, -math.inf, -math.inf]  # initializing U
probabilities = np.zeros((100, 1), dtype="float")
p = 0
sigam = 0 
selection_rate = 0.8
chromosome_length = 16
mutation_rate = 0.2
crossover_rate = 0.8
elitism_rate = 0.5
generations = 100 # 
parent_1 = np.zeros((1, 4), dtype="float")
parent_2 = np.zeros((1, 4), dtype="float")
index_1 = 0 # index pf parent 1
index_2 = 0 # index pf parent 2
Offspring_1 = np.zeros((1, 4), dtype="float")
Offspring_2 = np.zeros((1, 4), dtype="float")
population_size = 100
elite_count = 0
temporary_fitness = 0

# DE variables and constants
a_p = np.zeros((1, 4), dtype="float") # random choices in X
b_p = np.zeros((1, 4), dtype="float") # random choices in X
c_p = np.zeros((1, 4), dtype="float") # random choices in X
p_c = 0.5 # crossover probability
r_p = 0 # a random variable selected using np.uniform(0, 1)
Beta = 0 


# importing images
image_1 = cv2.imread("E:\\ali_project\\synpic15935_1.jpg", cv2.IMREAD_GRAYSCALE)/25.6
image_2 = cv2.imread("E:\\ali_project\\synpic28644_1.jpg", cv2.IMREAD_GRAYSCALE)/25.6
image_3 = cv2.imread("E:\\ali_project\\synpic51882_1.jpg", cv2.IMREAD_GRAYSCALE)/25.6

# Glaobal Mean
M_1 = np.mean(image_1) # global mean for image_1
M_2 = np.mean(image_2) # global mean for image_2
M_3 = np.mean(image_3) # global mean for image_3


# computing the gray level mean
def compute_mean(image, i, j):
    half_n = 3 // 2
    window = image[max(0, i-half_n):min(image.shape[0], i+half_n+1), 
                   max(0, j-half_n):min(image.shape[1], j+half_n+1)]
    return float(np.mean(window))#/27.5

# Computing the standard devation 
def compute_sigma(image, i, j):
    half_n = 3 // 2
    window = image[max(0, i-half_n):min(image.shape[0], i+half_n+1), 
                   max(0, j-half_n):min(image.shape[1], j+half_n+1)]
    mean = np.mean(window)
    return np.sqrt(np.mean((window - mean) ** 2))

# computing the transformed image
def transform_image(image, a, b, c, k, M):
    transformed_image = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            m_ij = compute_mean(image, i, j)
            sigma_ij = compute_sigma(image, i, j)
            f_ij = image[i, j]#/26.5
            g_ij = k * (M / (sigma_ij + b)) * (f_ij - c * m_ij) + m_ij ** a
            transformed_image[i, j] = g_ij
    # transformed_image = np.clip(transformed_image, 0, 255).astype(np.uint8)
    return transformed_image

# calculating the Entropy (Beta_g)
def calculate_entropy(image):
    # Calculate the histogram
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    # Normalize the histogram to get probabilities
    hist = hist / hist.sum()
    # Calculate the entropy
    entropy = -np.sum(hist * np.log2(hist + 1e-10))  # Adding a small value to avoid log(0)
    return entropy

#################################################################### create the enhanced image
# Enhanced_image = transform_image(image_1, 2.50, 3.0, 0.9891, 4.0, M_1 * 25.5)#/25.5
# Enhanced_image = np.clip(Enhanced_image, 0, 255).astype(np.uint8)
# cv2.imwrite("E:\\ali_project\\Enhanced_image.jpg", Enhanced_image)
# cv2.imshow ("E:\\ali_project\\Enhanced_image.jpg", Enhanced_image)

# calculating the MSE
def MSE_calculation(image, Enhanced_image):
    MSE = 0
    for i in range(Enhanced_image.shape[0]):
        for j in range(Enhanced_image.shape[1]):
            MSE += math.sqrt(abs(image[i, j] - Enhanced_image[i, j]))
    MSE /= (256 * 256)
    return MSE

####### PSNR value
# ro = 10 * math.log10(((L - 1) ** 2)/MSE) # ro is PSNR
# L = Enhanced_image.max() # max pixel intensity value in g(i, j)
def max_intensity(image):
    L = image.max()
    return L

########## calculating the number of edge pixels 
def edge_pixels(Enhanced_image):
    sobel_x = cv2.Sobel(Enhanced_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(Enhanced_image, cv2.CV_64F, 0, 1, ksize=3)
    sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    S_f = np.mean(sobel_magnitude)
    _, edge_pixels = cv2.threshold(sobel_magnitude, S_f, 255, cv2.THRESH_BINARY)
    n_edges = np.sum(edge_pixels > 0)
    return n_edges

########### n_edges = edge_pixels(enhanced_image)
# print("the number of edges are: ", n_edges)
def foreground_pixels(enhanced_image):
    threshold_value = threshold_otsu(enhanced_image)
    FI = np.sum(enhanced_image > threshold_value)
    return FI

# Evaluation Function or fitness function
def evaluation_function(Enhanced_image, original_image):
    number_of_edges = edge_pixels(Enhanced_image)                                           # calculating the number of edges
    # print(f'number_of_edges is {number_of_edges}')
    FI = foreground_pixels(Enhanced_image)                                                  # calculating the number of pixels belonging to the foreground object
    # print(f'FI is {FI}')
    mse = MSE_calculation(original_image, Enhanced_image)                                   # calculating the Mean Square Error
    # print(f'mse is {mse}')
    L = max_intensity(Enhanced_image)                                                       # calculating the maximum intensity valeu
    # print(f'L is {L}')
    RO = 10 * math.log10(((L - 1) ** 2)/mse) # ro is PSNR                                   # peak signal to noise ratio
    # print(f'RO is {RO}')
    Beta_g = calculate_entropy(Enhanced_image)                                              # calculating the entropic measure  
    # print(f'Beta_g is {Beta_g}')

    e_f = 1 - math.exp(-RO/100) + ((number_of_edges + FI) / (256 * 256)) + (Beta_g / 8)     # calculating the fitness which is between 0 to 4
    # print(f'fitness is : {e_f}')
    return e_f

while FCR < 1: # back before FCR_max was 200 but here we use 20 020
    for p in range(POP):

        # creating random parameters
        X[p, 0] = random.uniform(lb_a, ub_a) # creating random variables a
        L[0] = min(L[0], X[p, 0])
        U[0] = max(U[0], X[p, 0])
        X[p, 1] = random.uniform(lb_b, ub_b) # creating random variables b
        L[1] = min(L[1], X[p, 1])
        U[1] = max(U[1], X[p, 1])
        X[p, 2] = random.uniform(lb_c, ub_c) # creating random variables c
        L[2] = min(L[2], X[p, 2])
        U[2] = max(U[2], X[p, 2])
        X[p, 3] = random.uniform(lb_k, ub_k) # creating random variables k
        L[3] = min(L[3], X[p, 3])
        U[3] = max(U[3], X[p, 3])

        # creating enhanced image and calculating the fitness
        G_ij = transform_image(image_1, X[p,0], X[p,1], X[p,2], X[p,3], M_1)
        G_ij = np.clip(G_ij, 0, 255).astype(np.uint8)
        fitness[p] = evaluation_function(G_ij, image_1)

        if fitness[p] > fitness_global_best:
            fitness_global_best = fitness[p]
            X_best = X[p]
    FCR += 1

best_random_image_GENETIC = transform_image(image_1, X_best[0], X_best[1], X_best[2], X_best[3], M_1)
best_random_image_GENETIC = np.clip(best_random_image_GENETIC, 0, 255).astype(np.uint8)
cv2.imwrite("E:\\ali_project\\best_random_image_GENETIC.jpg", best_random_image_GENETIC)
cv2.imshow ("E:\\ali_project\\best_random_image_GENETIC.jpg", best_random_image_GENETIC)
print(f'\nBest random fitness is: {fitness_global_best}\nand best parameter valeus are: {X_best}\nPhase one has finished successfully')


FCR = 0
# X's shape is "100 x 4" and global best's shape is "1 x 4" 
global_best = X_best # the best set of parameters 

# GENETIC algorithm starts here

# roulette wheel function
def roulette_wheel(set, fitness_):
    sum_fitness = sum(fitness_)
    probabilities = [ F / sum_fitness for F in fitness_]
    selected = random.choices(set, weights=probabilities, k=int(0.01 * len(set)))
    # selected = np.random.choice(len(set), p=probabilities)
    return selected

# crossover functoin
def single_point_crossover(parent_one, parent_two):
    parent_x = np.zeros((1, 4), dtype="float")
    parent_y = np.zeros((1, 4), dtype="float")

    random_number = random.randint(1, 3)
    parent_x = parent_one[:random_number] + parent_two[random_number:]
    parent_y = parent_one[:random_number] + parent_two[random_number:]
    return parent_x, parent_y

# mutation function
def mutation_funciton(previous_X, Upper, Lower):
    X_new = np.zeros((1, 4), dtype="float")
    result_1 = np.random.normal(size=4)
    result_2 = np.subtract(Upper, Lower)
    result_3 = np.dot(0.1, result_2)
    X_new = previous_X + np.dot(result_3, result_1)
    return X_new # X_new dimension is [1 x 4]

# limitation function
def limitation_function(previous_X):
    # limitiing the parameters in X 
    previous_X[0, 0] = np.clip(X[p,0], lb_a, ub_a)
    previous_X[0, 1] = np.clip(X[p,1], lb_b, ub_b)
    previous_X[0, 2] = np.clip(X[p,2], lb_c, ub_c)
    previous_X[0, 3] = np.clip(X[p,3], lb_k, ub_k)
    return previous_X

def L_and_U(previous_X, L, U):
        L[0] = min(L[0], previous_X[0, 0])
        U[0] = max(U[0], previous_X[0, 0])
        L[1] = min(L[1], previous_X[0, 1])
        U[1] = max(U[1], previous_X[0, 1])
        L[2] = min(L[2], previous_X[0, 2])
        U[2] = max(U[2], previous_X[0, 2])
        L[3] = min(L[3], previous_X[0, 3])
        U[3] = max(U[3], previous_X[0, 3])
        return L, U


# main loop
while FCR < FCR_max:
    print("FCR IS: ", FCR)
    if FCR > 0:
        X = new_X
        fitness = new_fitness
        new_fitness = []
        new_X = []

    for p in range(POP):
        print("number of iteration: ", p)
        if p < 1:
            elite_count = int(elitism_rate * population_size)
            elites = sorted(X, key=lambda x: fitness[np.where(X == x)[0][0]], reverse=True)[:elite_count]
            new_X.extend(elites)
            elites_fitness = sorted(fitness, reverse=True)[:elite_count]
            new_fitness.extend(elites_fitness)
            
            ## SELECTION PHASE ##
            while np.array_equal(parent_1, parent_2):
                parent_1 = roulette_wheel(X, fitness)
                index_1 = np.where(np.all(X == parent_1, axis=1))[0][0]
                parent_2 = roulette_wheel(X, fitness)
                index_2 = np.where(np.all(X == parent_2, axis=1))[0][0]
            # remember to set the parents to zero
        
        ## CROSSOVER PHASE ##
        if random.random() <= crossover_rate:
            if len(new_X) < population_size:
                # creating the offsprings
                Offspring_1, Offspring_2 = single_point_crossover(parent_1, parent_2)
                
                # mutating, limiting, and inserting offspring_1
                Offspring_1 = mutation_funciton(Offspring_1, L, U)
                Offspring_1 = limitation_function(Offspring_1)
                L, U = L_and_U(Offspring_1, L, U)
                new_X.append(Offspring_1)
                
                # creating enhanced image and calculating the fitness of the offspring_1
                G_ij = transform_image(image_1, Offspring_1[0, 0], Offspring_1[0, 1], Offspring_1[0, 2], Offspring_1[0, 3], M_1)
                G_ij = np.clip(G_ij, 0, 255).astype(np.uint8)
                temporary_fitness = evaluation_function(G_ij, image_1)
                new_fitness.append(temporary_fitness)
                if temporary_fitness > fitness_global_best:
                    fitness_global_best = temporary_fitness
                    X_best = Offspring_1
                
                
                if len(new_X) < population_size:
                    # mutating, limiting, and inserting offspring_2
                    Offspring_2 = mutation_funciton(Offspring_2, L, U)
                    Offspring_2 = limitation_function(Offspring_2)
                    L, U = L_and_U(Offspring_2, L, U)
                    new_X.append(Offspring_2)
                    
                    # creating enhanced image and calculating the fitness of the offspring_2
                    G_ij = transform_image(image_1, Offspring_2[0, 0], Offspring_2[0, 1], Offspring_2[0, 2], Offspring_2[0, 3], M_1)
                    G_ij = np.clip(G_ij, 0, 255).astype(np.uint8)
                    temporary_fitness = evaluation_function(G_ij, image_1)
                    new_fitness.append(temporary_fitness)
                    if temporary_fitness > fitness_global_best:
                        fitness_global_best = temporary_fitness
                        X_best = Offspring_2


# visualization the results
if X_best[2] > 0.955:
    result_image_GENETIC = transform_image(image_1, X_best[0], X_best[1], X_best[2], X_best[3], M_1 * 1.9)
elif X_best[2] < 0.96 and X_best > 0.93:
    result_image_GENETIC = transform_image(image_1, X_best[0], X_best[1], X_best[2], X_best[3], M_1 * 1.2)
elif X_best[2] < 0.93 and X_best > 0.89:
    result_image_GENETIC = transform_image(image_1, X_best[0], X_best[1], X_best[2], X_best[3], M_1 * 0.9)
else :
    result_image_GENETIC = transform_image(image_1, X_best[0], X_best[1], X_best[2], X_best[3], M_1)#/25.5

result_image_GENETIC = np.clip(result_image_GENETIC, 0, 255).astype(np.uint8)
cv2.imwrite("E:\\ali_project\\result_image_GENETIC.jpg", result_image_GENETIC)
cv2.imshow ("E:\\ali_project\\result_image_GENETIC.jpg", result_image_GENETIC)
print(f'\nFinal result --->>>\nBest fitness is: {fitness_global_best}\nand best parameters are: {X_best}') 
