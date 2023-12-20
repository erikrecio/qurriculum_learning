#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import cma
from scipy.optimize import minimize
from qibo.symbols import X, Y, Z, I
from qibo import hamiltonians
from qibo import models, gates, hamiltonians
import tensorflow as tf
import qibo
qibo.set_backend("numpy")
from shapely.geometry import Polygon, Point


def main(training_data):
    def generar_punto_en_region(region_poly):
        while True:
            min_x, min_y, max_x, max_y = region_poly.bounds
            x = np.random.uniform(min_x, max_x)
            y = np.random.uniform(min_y, max_y)
            punto = Point(x, y)
            if region_poly.contains(punto):
                return x, y
            
    def labeling(x, y):
        # Definir las coordenadas de los puntos de cada región
        region1_coords = [(-2, 1), (2, 1), (0, -1)]
        region2_coords = [(0, -1), (3, -4), (4, -4), (4, 3)]
        region3_coords = [(0, -1), (-3, -4), (-4, -4), (-4, 3)]
        region4_coords = [(-2, 1), (2, 1), (4, 3), (4, 4), (-4, 4), (-4, 3)]
        region5_coords = [(-3, -4), (0, -1), (3, -4)]
        
        # Crear objetos Polygon para cada región
        region1_poly = Polygon(region1_coords)
        region2_poly = Polygon(region2_coords)
        region3_poly = Polygon(region3_coords)
        region4_poly = Polygon(region4_coords)
        region5_poly = Polygon(region5_coords)
        
        punto = Point(x, y)
        if region1_poly.contains(punto):
            return 3
        elif region2_poly.contains(punto):
            return 1
        elif region3_poly.contains(punto):
            return 2
        elif region4_poly.contains(punto):
            return 0
        elif region5_poly.contains(punto):
            return 0
        else:
            return None # Si el punto no está en ninguna región
    
    def ground_state(j1, j2):
        symbolic_expr = Z(0) + Z(1) + Z(2) + Z(3) + Z(4) + Z(5) + Z(6) + Z(7)
        symbolic_expr -= j1*(X(0)*X(1) + X(1)*X(2) + X(2)*X(3) + X(3)*X(4) + X(4)*X(5) + X(5)*X(6) + X(6)*X(7) + X(7)*X(0))
        symbolic_expr -= j2*(X(0)*Z(1)*X(2) + X(1)*Z(2)*X(3) + X(2)*Z(3)*X(4) + X(3)*Z(4)*X(5) + X(4)*Z(5)*X(6) + X(5)*Z(6)*X(7) + X(6)*Z(7)*X(0) + X(7)*Z(0)*X(1))        
        hamiltonian = hamiltonians.SymbolicHamiltonian(form=symbolic_expr)
        ground_state = hamiltonian.ground_state()
        return ground_state

    gs_list = []
    label_list = []
    j1_list = np.random.uniform(-4, 4, 15)
    j2_list = np.random.uniform(-4, 4, 15)
    
    # Definir las coordenadas de los puntos de cada región
    region1_coords = [(-2, 1), (2, 1), (0, -1)]
    region2_coords = [(0, -1), (3, -4), (4, -4), (4, 3)]
    region3_coords = [(0, -1), (-3, -4), (-4, -4), (-4, 3)]
    region4_coords = [(-2, 1), (2, 1), (4, 3), (4, 4), (-4, 4), (-4, 3)]
    region5_coords = [(-3, -4), (0, -1), (3, -4)]
    
    # Crear objetos Polygon para cada región
    region1_poly = Polygon(region1_coords)    
    region2_poly = Polygon(region2_coords)
    region3_poly = Polygon(region3_coords)
    region4_poly = Polygon(region4_coords)
    region5_poly = Polygon(region5_coords)
    
        
    for i in range(training_data):
        print(labeling(j1_list[i], j2_list[i]))
        label_list.append(labeling(j1_list[i], j2_list[i]))
        
    for k in range(training_data):        
        gs_list.append(ground_state(j1_list[k], j2_list[k]))
    
    np.savetxt("J1coef_j1j2_15_Training_Set", [j1_list], newline='')
    np.savetxt("J2coef_j1j2_15_Training_Set", [j2_list], newline='')
    np.savetxt("LABELS_15_Training_Set", [label_list], newline='')
    
    np.save('training_set_15examples', gs_list)
    # gs_list = np.load('test_set_1000examples.npy', allow_pickle=True)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_data", default=15, type=int)
    args = parser.parse_args()
    main(**vars(args))
