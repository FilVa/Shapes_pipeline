# Shapes_pipeline

Code implementation for pipeline in paper:
F. Valdeira, R. Ferreira, A. Micheletti and C. Soares, "From Noisy Point Clouds to Complete Ear Shapes: Unsupervised Pipeline," in IEEE Access, vol. 9, pp. 127720-127734, 2021, 
https://ieeexplore.ieee.org/document/9534754?source=authoralert

# Run code

1. In folder 'Data/Original_data' place the 3D mesh files you wish to submit to the pipeline process. Files shoud be named with ID numbers of each shape.
2. In folder 'Data/Template' place the 3D mesh file you desire to use as template.
3. In folder 'PCA_data' a set of registered shapes (also registered with respect to the template) to be used for the PCA kernel.
4. Run main_script.py to execute the pipeline.
5. The results for each step will be placed in the following folders 'Data/Step_1_results','Data/Step_2_results','Data/Step_3_results'
