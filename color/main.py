from deap import base, creator, tools, algorithms
from tensorflow.keras.models import load_model
import numpy as np
import random
import matplotlib.pyplot
import pickle
import os.path
import tensorflow as tf

model = load_model('cifar10-model-v1.h5')

def preprocess_image_color(image):
	# convert [0, 255] range integers to [0, 1] range floats
	image = image / 255.0
	return image

# individuo es un array
def evaluateInd(individual):
	# Evaluamos con NN
	image = chromosome2img(individual, (32,32,3))
	image = preprocess_image_color(image)
	prediction = model.predict(image.reshape(-1, *image.shape))[0]

	diff_pixels = 0
	for i in range(int(len(individual)/3)-1):
		diff_pixels += abs(individual[i]-individual[i+3]) + abs(individual[i+1]-individual[i+4]) + abs(individual[i+2]-individual[i+5])
	diff_pixels = diff_pixels/(255*1000)

	#quality = prediction[0] #0 es airplane
	suma = 0
	for x in range(10):
		suma += prediction[x]
	suma -= prediction[2]
	quality = prediction[2] - suma - diff_pixels #2 es bird
	return quality,

def chromosome2img(chromosome, img_shape):
	"""
	First step in GA is to represent the input in a sequence of characters.
	The encoding used is value encoding by giving each gene in the chromosome 
	its actual value.
	"""
	img_arr = np.reshape(a=chromosome, newshape=img_shape)
	return img_arr


POP_SIZE = 60
IND_SIZE = 32*32*3 # aca va 28*28 o 32*32*3 o tamano de la imagen

creator.create("FitnessMax", base.Fitness, weights=(1.0,))

creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()

#Solo muta el cuadrado de 20*20 interno de la solucion
# def randomInitCenter():
# 	individual = np.zeros(IND_SIZE, dtype=int)
# 	for i in range(28*4,IND_SIZE-28*4):
# 		if (i % 28) >= 4 and (i % 28) <= 24:
# 			if random.random() > 0.5:
# 				individual[i] = random.randint(1, 255)
# 			else:
# 				individual[i] = 0
# 	return individual

toolbox.register("attr_int_0_255", random.randint, 0, 255) #attr entre 0 y 255
# Generador de atributos
# toolbox.register("indices", random.sample, range(IND_SIZE), IND_SIZE)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int_0_255, n=IND_SIZE)
#toolbox.register("individual", tools.initIterate, creator.Individual, randomInitCenter)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluateInd)


def mutFlipBitCustom(individual, indpb):
	for i in range(28*4,len(individual)-28*4):
		if (i % 28) >= 4 and (i % 28) <= 24:
			if random.random() < indpb: #prob de mutar gen
				if random.random() > 0.5: #mut1 o mut2
					if random.random() > 0.5: #mut1:negro o gris
						individual[i] = random.randint(1, 255)
					else:
						individual[i] = 0
				else: #mut2 promedio de vecinos
					individual[i] = round((individual[i-1] + individual[i+1] + individual[i+29] + individual[i+28] + individual[i+27] + individual[i-29] + individual[i-28] + individual[i-27]) / 8)
	return individual,

def mutColor(individual, indpb):
	if random.random() < 0.5:
		return tools.mutUniformInt(individual=individual, indpb=indpb, low=0, up=255)
	else:
		for i in range(len(individual)):
			if random.random() < indpb:
				#mutacion a promedio de vecinos
				#96 = 32*3
				if i>=96 and i<=len(individual)-96 and i % 96 != 0 and i % 96 != 95:
					individual[i] = round((individual[i-3] + individual[i+3] + individual[i+29*3] + individual[i+28*3] + individual[i+27*3] + individual[i-29*3] + individual[i-28*3] + individual[i-27*3]) / 8)
		return individual,

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", mutColor, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=5)
#############

def main(checkpoint=None, output=''):
	if(checkpoint):
		with open(checkpoint, "rb") as cp_file:
			cp = pickle.load(cp_file)
		pop = cp["population"]
	else:
		pop = toolbox.population(n=POP_SIZE) # Lista con POP_SIZE individuos
	# CXPB es probabilidad de cruzamiento
	CXPB = 0.6
	# MUTPB as la probability de mutacion de un individuo
	MUTPB = 0.2
	NUM_GEN = 900

	hof = tools.HallOfFame(1)
	stats = tools.Statistics(lambda ind: ind.fitness.values)
	stats.register("avg", np.mean)
	stats.register("std", np.std)
	stats.register("min", np.min)
	stats.register("max", np.max)

	
	pop, log = algorithms.eaSimple(pop, toolbox, cxpb=CXPB, mutpb=MUTPB, ngen=NUM_GEN, 
	                               stats=stats, halloffame=hof, verbose=True)
	# it = 0
	# while it < NUM_GEN:
	# 	pop, log = algorithms.eaSimple(pop, toolbox, cxpb=CXPB, mutpb=MUTPB, ngen=10, 
	#                                stats=stats, halloffame=hof, verbose=True)
	# 	it += 10										 
	# 	for solution in pop:
	# 		for i in range(28,len(solution)-28):
	# 			if solution[i] != 0  and  0 == solution[i-1] and 0 == solution[i+1] and 0 == solution[i-28] and 0 == solution[i+28]:
	# 				solution[i] = 0
	# 	print('it:', it)
	
	cp = dict(population=pop)
	with open('pop_cp_'+output+'.pickle', "wb") as cp_file:
		pickle.dump(cp, cp_file)
	
	best_solution = tools.selBest(pop, 1, fit_attr='fitness')
	best_solution_img = chromosome2img(best_solution, (32,32,3))
	print("best_solution_img", best_solution_img)
	best_solution_img = preprocess_image_color(best_solution_img)
	predict = model.predict(best_solution_img.reshape(-1, *best_solution_img.shape))[0]
	print('predict', predict)
	#best_solution_img = best_solution_img * 255.0
	matplotlib.pyplot.imsave(output+'.png', best_solution_img)


input1 = 'test_100_fitness'
output1 = 'test_1000_fitness'
if os.path.isfile('pop_cp_'+input1+'.pickle'):
	main('pop_cp_'+input1+'.pickle', output=output1)
else:
	main(output=output1)




