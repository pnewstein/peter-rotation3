Code from my rotatation in the sylwestrak lab. 
There are two modules ```morpho.py``` which contains code that reads swc files exported by the fiji plugin SNT
```electro.py``` that reads abf files, does some simple analysis, and creates the figures I used in my rotation talk
```electro.py``` depends on tha allen institute sdk and ```morpho.py``` depends on navis. I was unable to create a single 
python environment with both dependencies installed, so some sort of venv management is neccisary to run this code.
I used two conda environments
