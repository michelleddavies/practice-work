'''

This is a Python class library designed to create simulations in Evolutionary Algorthims and game theory.

Copyright: Michelle Davies (mdd94, Cornell ECE '22), June 2022


'''

import math
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sympy
import itertools

class game_theory:
	"""
	This class implements some of the basic methods to implement game theory examples.

	The games supported in this class are:
		1. TIT FOR TAT
		2. Battle of the Sexes
		3. Hawk-Dove Game / Game of Chicken
		4. Prisoner's Dilemma
		5. Rock, Paper, Scissor (Scissor, Paper, Rock)
		6. MP Game
		7. Chain Store Game
		8. Random
	And other multiplayer custom levels.

	"""
	## Class variables
	g_mat = []
	games = ["Custom", "TIT FOR TAT", "Battle of the Sexes", "Hawk-Dove Game / Game of Chicken", "Prisoner's Dilemma", "Rock, Paper, Scissor (Scissor, Paper, Rock)", "Matching Pennies / Zero-Sum Game", "Chain Store Game", "Random"]
	current_game = ""
	cg_int = 0
	dimensions = 0
	msne_list = []
	psne_list = []
	strict_ne_list = []
	ess_list = []
	## use an init function to define the mode of game and construct the game matrix.
	print("Ready to create Game Space. Please note that dimension-picking is only supported in random and custom mode.")
	print("Game Menu: \n The games supported in this class are: \n1. TIT FOR TAT \n2. Battle of the Sexes \n3. Hawk-Dove Game / Game of Chicken \n4. Prisoner's Dilemma \n5. Rock, Paper, Scissor (Scissor, Paper, Rock) \n6. Matching Pennies / Zero-Sum Game \n7. Chain Store Game \n8. Random \n0. Custom \n\n")
	def __init__(self, game, dimensions=2):
		# determine the numbers in the matrix based on the game and dimensions
		self.dimensions = dimensions
		game = game % 9
		self.current_game = self.games[game]
		self.cg_int = game
		print(f'=== You have selected game {game}: {g} ===')
		if (game == 1): # TIT FOR TAT
			self.g_mat = [[(3,3), (0,5)], [(5,0), (1,1)]]
			self.dimensions = 2
		elif (game == 2): # Battle of the Sexes
			self.g_mat = [[(3,2), (1,1)], [(1,1), (2,3)]]
			self.dimensions = 2
		elif (game == 3): # Hawk-Dove Game / Game of Chicken
			self.g_mat = [[(-25,-25), (50,0)], [(0,50), (15,15)]]
			self.dimensions = 2
		elif (game == 4): # Prisoner's Dilemma
			self.g_mat = [[(3,3), (0,5)], [(5,0), (1,1)]]
			self.dimensions = 2
		elif (game == 5): # Rock, Paper, Scissor (Scissor, Paper, Rock)
			self.g_mat = [[(1,1), (2,0), (0,2)], [(0,2), (1,1), (2,0)], [(2,0), (0,2), (1,1)]]
			self.dimensions = 3
		elif (game == 6): # MP / Zero-Sum Game
			self.g_mat = [[(1,-1), (-1,1)], [(-1,1), (1,-1)]]
			self.dimensions = 2
		elif (game == 7): # Chain Store Game
			self.g_mat = [[(2,2), (-1,-1)], [(0,4), (0,4)]]	
			self.dimensions = 2
		elif (game == 8): # random 
			self.g_mat = [[(random.randint(0, 25), random.randint(0, 25)) for c in range(dimensions)] for r in range(dimensions)]
		else: # custom
			# gather user input for numbers to use the build the matrix
			numbers = []
			print(f'Welcome to custom mode. \nPlease input {dimensions} pairs of numbers that you want to use in the matrix.\n\n')
			for num in range(dimensions):
				t = dimensions-num
				print(f'{t} pairs left to provide.')
				uint1 = int(input("Please provide the first number in the pair: "))
				uint2 = int(input("Please provide the second number in the pair: "))
				uint = tuple(uint1, uint2)
				numbers.append(uint)
			# build circulant matrix with given values
			self.g_mat = []
			r = numbers[-1:] + numbers[:-1]
			for num in range(dimensions):
				r = r[-1:] + r[:-1]
				self.g_mat.append(r)
		return self.g_mat

	## Find PSNE
	def psne(self):
		# first, for each row (i.e set action from player 1), determine the action (column) for which player 2 yields the highest playoff.
		nash_e = []
		for row in self.g_mat:
			p2_payoffs = [t[1] for t in row]
			max_p2 = max(p2_payoffs)
			idx_max_p2 = p2_payoffs.index(max_p2)
			ne = row[idx_max_p2]
			nash_e.append(ne)
		# next, for each colum (i.e set action from player 2), determine the action (row) for which player 1 yields the highest playoff.
		mat_tr = np.transpose(self.g_mat)
		for row in mat_tr:
			p1_payoffs = [t[1] for t in row]
			max_p1 = max(p1_payoffs)
			idx_max_p1 = p1_payoffs.index(max_p1)
			ne = row[idx_max_p1]
			nash_e.append(ne)
		self.psne_list = nash_e
		return nash_e

	def __altBR(self, tlist):
		# return a list of actions which have an alt best reply in a given list of 2D tuples
		assert len(tlist) != 0
		# check for alt best replies in lists
		p1_acts = [t[0] for t in tlist]
		p2_acts = [t[1] for t in tlist]
		# assert same size
		if (abs(len(p1_acts)-len(p2_acts)) != 0):
			for i in range(abs(len(p1_acts)-len(p2_acts))):
				if (len(p1_acts) < len(p2_acts)):
					p1_acts.append(None)
				else: # len(p1_acts) > len(p2_acts)
					p2_acts.append(None)
		# compare by index
		l = [t for t in tlist if p1_acts.count(t[0]) > 1 or p2_acts.count(t[1]) > 1]
		return l


	## Find Nash Equillibria, specify which are strict
	def strict_ne(self):
		nes = self.psne_list + self.msne_list
		sne = dict()
		if (len(nes) == 0):
			return None
		# make a comprehensive list of tuples where the first element of one tuple is not the first element of any other tuple.
		altBRs = self.__altBR(nes)
		sl = [t for t in nes if t not in altBRs]
		sne['strict_list'] = sl
		sne['strict_count'] = len(sl)
		self.strict_ne_list = sl
		return sne

	## Find MSNE
	def msne(self):
		gm = self.g_mat
		ms_ne = []
		# algebra
		p, q = sympy.symbols('p q') # probablities
		A, B, C, D = sympy.symbols('A B C D') # actions
		sigma1 = p*A + (1-p)*B
		sigma2 = q*C + (1-q)*D
		# get all combinations of 2 by 2 matrices in the game space
		idxs = [i for i in range(self.dimensions)]
		combos_p1 = itertools.combinations(idxs, 2)
		combos_p2 = itertools.combinations(idxs, 2)
		# for each possible combo of 2x2 matrices, calculate msne, add to list
		for c1 in combos_p1:
			for c2 in combos_p2:
				# get 2x2 matrix
				mtrx = [[gm[c1[0]], gm[c1[1]]], [gm[c2[0]], gm[c2[1]]]]
				# substitute values
				exp1_p1 = sigma1.subs([(A, mtrx[0][0][0]), (B, mtrx[0][1][0])])
				exp2_p1 = sigma1.subs([(A, mtrx[1][0][0]), (B, mtrx[1][1][0])])
				exp1_p2 = sigma2.subs([(C, mtrx[0][0][1]), (D, mtrx[0][1][1])])
				exp2_p2 = sigma2.subs([(C, mtrx[1][0][1]), (D, mtrx[1][1][1])])
				# equations
				player1 = sympy.Eq(exp1_p1, exp2_p1)
				player2 = sympy.Eq(exp1_p2, exp2_p2)
				# inequalities
				BR_player1 = sympy.solve(player1, p)
				BR_player2 = sympy.solve(player2, q)
				# check if the probabilities p, q are valid (within [0,1]). If they are, calculate the BR
				pProb = BR_player1[p]
				qProb = BR_player2[q]
				if (pProb <= 1 and pProb >= 0):
					# find & format mixed strategy to add to list
					br_a = [f'A is Best Reply when p < {pProb}', (A, "*")]
					br_b = [f'B is Best Reply when p > {pProb}', (B, "*")]
					br_mp = [(sigma1.subs(p,pProb), C), (sigma1.subs(p,pProb), D)]
					t = {'matrix':np.array(mtrx), 'first_act_BR':br_a, 'second_act_BR':br_b, 'msne':br_mp}
					ms_ne.append(t)
				else:
					# add BR to list
					play = A if pProb > 0 else B
					br = ['There exists a single BR.', (play, "*")]
					t = {'matrix':np.array(mtrx), 'act_BR':br, 'msne':None}
					ms_ne.append(t)
				if (qProb <= 1 and qProb >= 0):
					# find & format mixed strategy to add to list
					br_c = [f'C is Best Reply when q < {qProb}', ("*", C)]
					br_d = [f'D is Best Reply when q > {qProb}', ("*", D)]
					br_mq = [(A, sigma2.subs(q,qProb)), (B, sigma2.subs(q,qProb))]
					t = {'matrix':np.array(mtrx), 'first_act_BR':br_c, 'second_act_BR':br_d, 'msne':br_mq}
					ms_ne.append(t)
				else:
					# add BR to list
					play = C if qProb > 0 else D
					br = ['There exists a single BR.', ("*", play)]
					t = {'matrix':np.array(mtrx), 'act_BR':br,'msne':None}
					ms_ne.append(t)
		self.msne_list = ms_ne
		return ms_ne



class monte_carlo:
	"""
	Monte Carlo Programming Assignment, ECE 4271 Spring 2022
	Michelle Davies (mdd94) Monte Carlo - Metropolis Implementation

	Now for the Monte Carlo-type approach to computing Ep(f). I’d like you to construct code that implements a Metropolis-type Markov chain on S. Leave the spike parameter and number of iterations as user inputs. Initialize your iteration at a uniformly randomly chosen (m,n) ∈ S. Let Xk be the state of the Markov chain at iteration time k > 0. You’ll choose Xk+1:

	Based on the lab handout, this class contains my solution in Python, and then I will have a file mdd94_monte_carlo_test.py to test this class and gather data compared to the SLLN approach.

	"""

	def __init__(self, sample_size=10000):
		# generate random samples
		self.S = set()
		for i in range(sample_size):
			m = random.randint(0,99)
			n = random.randint(0,99)
			tup = (m,n)
			self.S.add(tup)
		self.S_size = len(self.S)
		self.P_Xk = dict()
		self.theory_Epf = 100

	## get functions
	def getSampleSz(self):
		return self.S_size

	def getS(self):
		return self.S

	def getP_Xk(self):
		return self.P_Xk

	def get_mc_time_avg(self, N=10000):
		return self.set_mc_time_avg(N)

	def get_qfunct(self, m, n, sigma=1):
		c = abs(m - 50)**sigma
		print("c = " + str(c))
		d = abs(n - 50)**sigma
		print("d = " + str(d))
		answer = 1 / (1 + c + d)
		print("Prob: " + str(answer))
		return answer

	def get_probability(self, m, n, m2, n2):
		sumq = 0
		for i in range(m2):
			for j in range(n2):
				sumq += self.get_qfunct(i,j)
		p = self.get_qfunct(m, n) / sumq
		return p

	## set functions (calculating things)

	def set_mc_time_avg(self, N=10000):
		#set self.mc_time_avg
		time_average = 0
		setP = self.P_Xk # want to preserve the self.P_Xk property in case of a bug
		print(str(setP))
		target_set = setP[N] # get the set of states with N runs
		print(str(target_set))
		for element in target_set:
			print("element = " + str(element))
			fsum = element[0]
			prob = element[1]
			time_average += (fsum[0] + fsum[1])
			print("time_average = " + str(time_average))
		#time_average = (1/N) * sum(target_set)
		ans = round((1/len(target_set)) * time_average, 5)
		print("total time_average = " + str(ans))
		return ans

	def monte_carlo(self, runs=10000, arrayed_iter=((-1,1), (0,1), (1,1), (-1,0), (0,0), (1,0), (-1,-1), (0,-1), (1,-1))): #10000 runs by default, can adjust at runtime by defining parameter at function call in main program, and offsets for locations of nine nearest neighbors by default, cn pass a list of tuples with larger offsets
		# create a dictionary of Xks and neighbor history for the function to return
		monte_run = dict()
		states = set()
		# initialize step - choose (m,n) from S
		tuple_options = list(self.S)
		#print(str(tuple_options))
		Xk = random.choice(tuple_options)
		print(str(Xk))
		prob = 1
		states.add((Xk, prob)) # state, probability
		for r in range(runs):
			print("\nNew Run:\n")
			Y_kplus1 = []
			# propose step - choose Yk+1 uniformly over the nine nearest neighbors of Xk, including Xk itself.
			 # offsets for locations of nine nearest neighbors
			for i in arrayed_iter:
				mplus1 = (Xk[0]+i[0]) % 100
				nplus1 = (Xk[1]+i[1]) % 100
				#print(str(mplus1))
				#print(str(nplus1))
				Y_kplus1.append((mplus1, nplus1))
			monte_run[Xk] = Y_kplus1
			# accept/reject step for all neighbors:
			for neighbor in Y_kplus1:
				print("Neighbor: " + str(neighbor))
				print("Current: " + str(Xk))
				print("Current @ 0: " + str(Xk[0]))
				print(type(Xk[0]))
				print("Current @ 1: " + str(Xk[1]))
				print(type(Xk[1]))
				a = self.get_qfunct(Xk[0], Xk[1])
				b = self.get_qfunct(neighbor[0], neighbor[1])
				if (b >= a): # q (Yk+1) ≥ q (Xk)
					Xk = neighbor
					prob = b/a  #b/a
				else: # q (Yk+1) < q (Xk)
					# Xk remains the same so no reason to update the Xk field to Xk = Xk
					prob = 1 - b/a  #1 - (b/a)
			# set property values
			states.add((Xk, prob))
		self.P_Xk[runs] = states
		return monte_run # just want to return the outcome Xk states and all associated neighbors for the function, the list of states that were actually selected for a run will become an element of the obj attribute P_Xk, the set P of all the Xk-values generated

	## Comparison to theory Epf
	def Epf__compare(self, N=10000):
		# return whether the theory is around the actual with margin of error (bool)
		mc_Epf = self.get_mc_time_avg(N)
		print("mc_Epf = " + str(mc_Epf))
		if (round(mc_Epf, -2) == 100): # within margin of error
			return True
		else: # not within margin of error
			return False

	## Comparison function to SLLN
	def slln_compare(self, spike, N=10000):
		# compare Monte Carlo - Metropolis Implementation to the example SLLN Implementation from the handout
		#spike = float(input(’Enter spikiness value: ’))
		#N = int(input(’Enter number of iterations: ’))
		def q(m,n):
			return 1/(1 + (abs(m - 50))**spike + (abs(n - 50))**spike)
		time_avg = 0
		for k in list(range(N)):
			m = random.randint(0,99)
			n = random.randint(0,99)
			print("pair val = ({one}, {two})".format(one=m, two=n))
			normalization = 0
			if (m == 0 and n != 0):
				for j in range(n):
					normalization += q(0,j)
			elif (n == 0 and m != 0):
				for i in range(m):
					normalization += q(i,0)
			elif (m != 0 and n != 0):
				for i in range(m):
					for j in range(n):
						normalization += q(i,j)
			else:
				normalization = q(m,n)
			print("normalization = {n}".format(n=normalization))
			time_avg = (k*time_avg + N*(m + n)*(q(m , n)/normalization))/(k + 1)
		print("Computed mean = " + str(time_avg))
		# get the Computed mean of Monte Carlo to compare
		# if time_avg > self.mc_time_avg, return -1 for "Monte Carlo avg is less than time_avg of SLLN, and the difference as a list
		# if time_avg < self.mc_time_avg, return 1 for "Monte Carlo avg is more than time_avg of SLLN, and the difference as a list
		# if time_avg == self.mc_time_avg, return 0 for "Monte Carlo avg is equal to time_avg of SLLN, and the difference as a list
		mc_timeavg = self.get_mc_time_avg(N)
		if (time_avg > mc_timeavg):
			return [-1, abs(time_avg - mc_timeavg)]
		elif (time_avg < mc_timeavg):
			return [1, abs(time_avg - mc_timeavg)]
		else: # time_avg == self.mc_time_avg
			return [0, abs(time_avg - mc_timeavg)]

	def graph_data(self, N, trial):
		# for varying runs, compare 3D P-histogram with a graph of p(m,n) vs. (m,n).
		P = self.P_Xk
		# for N = 10000
		p_N = P[N]
		print(str(p_N))
		Px_N = [tuple[0][0] for tuple in p_N]
		print(str(Px_N))
		Py_N = [tuple[0][1] for tuple in p_N]
		print(str(Py_N))
		prob_dist_N= [t[1] for t in p_N] 
		print(str(prob_dist_N))
		## 3d plot of points in P and p(m,n) vs. (m,n)
		fig = plt.figure(figsize=(10, 10), dpi=80)
		ax = fig.add_subplot(projection='3d')
		dx = np.ones(len(Px_N))
		dy = np.ones(len(Py_N))
		dz = np.multiply(np.zeros(len(prob_dist_N)), 0.25)
		ax.bar3d(Px_N, Py_N, prob_dist_N, dx, dy, dz)
		ax.set_xlabel('m')
		ax.set_ylabel('n')
		ax.set_zlabel('p(m, n)')
		file_graphb = "./graph_"+str(N)+"_("+str(trial)+")_bar3d.png" # file name
		plt.savefig(file_graphb)
		ax1 = fig.add_subplot(projection='3d')
		ax1.scatter(Px_N, Py_N, prob_dist_N)
		ax1.set_xlabel('m')
		ax1.set_ylabel('n')
		ax1.set_zlabel('p(m, n)')
		file_graph = "./graph_"+str(N)+"_("+str(trial)+")_plot.png" # file name
		plt.savefig(file_graph)
		#plt.show()
		return [file_graphb, file_graph]

