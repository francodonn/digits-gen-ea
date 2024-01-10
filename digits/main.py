from deap import base, creator, tools, algorithms
from tensorflow.keras.models import load_model
import numpy as np
import random
import matplotlib.pyplot

model = load_model('digits_con_ruido.h5')

def preprocess_image(img):
	# prepare pixel data
	# img = img.astype('float32')
	# img = img / 255.0
	# reshape into a single sample with 1 channel
	img = img.reshape(1,28,28,1)
	return img

# individuo es un array
def evaluateInd(individual):
	# Evaluamos con NN
	image = chromosome2img(individual, (28,28))
	image = preprocess_image(image)
	prediction = model.predict(image)[0]
	#quality = prediction[5]#queremos generar cincos
	suma = 0
	for x in range(11):
		suma += prediction[x]
	suma -= prediction[0]
	suma = suma / 10
	quality = prediction[0] - suma #queremos generar cincos
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
IND_SIZE = 28*28 # aca va 28*28 o 32*32*3 o tamano de la imagen

creator.create("FitnessMax", base.Fitness, weights=(1.0,))

creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()

toolbox.register("attr_int_0_1", random.randint, 0, 1) #attr entre 0 y 255
# Generador de atributos
# toolbox.register("indices", random.sample, range(IND_SIZE), IND_SIZE)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int_0_1, n=IND_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluateInd)

# Investigar mas
def mutFlipBitCustomNeighbors(individual, indpb):
	for i in range(1,len(individual)-1):
		if random.random() < indpb:
			if individual[i] != individual[i-1]  or  individual[i] != individual[i+1] or (i>28 and i<len(individual)-28 and (individual[i] != individual[i-28] or  individual[i] != individual[i+28])):
				individual[i] = type(individual[i])(not individual[i])
	return individual,

def mutFlipBitCustom(individual, indpb):
	for i in range(28*4,len(individual)-28*4):
		if (i % 28) >= 4 and (i % 28) <= 24:
			if random.random() < indpb:
				individual[i] = type(individual[i])(not individual[i])
	return individual,

toolbox.register("mate", tools.cxPartialyMatched)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.01)
toolbox.register("select", tools.selTournament, tournsize=5)
#############

def main():
	pop = toolbox.population(n=POP_SIZE) # Lista con POP_SIZE individuos
	# CXPB es probabilidad de cruzamiento
	CXPB = 0.6
	# MUTPB as la probability de mutacion de un individuo
	MUTPB = 0.2
	NUM_GEN = 10

	hof = tools.HallOfFame(1)
	stats = tools.Statistics(lambda ind: ind.fitness.values)
	stats.register("avg", np.mean)
	stats.register("std", np.std)
	stats.register("min", np.min)
	stats.register("max", np.max)

	pop, log = algorithms.eaSimple(pop, toolbox, cxpb=CXPB, mutpb=MUTPB, ngen=NUM_GEN, 
	                               stats=stats, halloffame=hof, verbose=True)

	best_solution = tools.selBest(pop, 1, fit_attr='fitness')
	for i in range(28,len(best_solution[0])-28):
		if best_solution[0][i] != best_solution[0][i-1]  and  best_solution[0][i] != best_solution[0][i+1] and best_solution[0][i] != best_solution[0][i-28] and best_solution[0][i] != best_solution[0][i+28]:
			best_solution[0][i] = best_solution[0][i-1]
	best_solution_img = chromosome2img(best_solution, (28,28))
	best_solution_img = best_solution_img * 255.0
	print("best_solution_img", best_solution_img)
	matplotlib.pyplot.imsave('a.png', best_solution_img, cmap='gray')

main()





