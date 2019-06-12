# bbob_numpy

This repository contains numpy implementations of the functions found in 

https://coco.gforge.inria.fr/downloads/download16.00/bbobdocfunctions.pdf

which allow accessing current x_opt and f_opt.

It's a bit work in progress, so I put some TODOs into the code. Feel free to extend or correct them. When plotting some 
functions in 2D they looked different than the plots in the pdf, so there may be some errors in implementation. 

## How to use
In order to create a suite with all objective functions in dimension 2 with 2 instances each do

```
test_suite_options = {'name': 'full',
                      'dim': [2],
                      'n_instances': 2}
test_suite = Suite(test_suite_options)

# query the optimal function value
for p in test_suite:
    print(p.f_obj.f_opt)
```