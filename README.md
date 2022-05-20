# ETKA
 
This repository contains the implementation of the Event-Triggered Kernel Adjustment for Gaussian Process Modelling. The corresponding publication will be reference here after acceptance.

The code builds on pytorch, gpytorch and botorch to offer a general-purpose framework for Gaussian Processes. It also includes the [Compositional Kernel Search](https://proceedings.mlr.press/v28/duvenaud13.html) and the [Adjusting Kernel Search](https://ieeexplore.ieee.org/abstract/document/9671767). The provided data is separated into synthetic data generated with [Nike's time series generator](https://github.com/Nike-Inc/timeseries-generator) and real-life data published by [Lloyd et al.](https://github.com/jamesrobertlloyd/gpss-research/tree/master/data/tsdlr-renamed).

The Research folder contains all data, data statistics and results used for the publication. It also includes a jupyter notebook with the code to reproduce all plots we show.

The implementation was done by [Jan David Hüwel](https://scholar.google.de/citations?hl=de&user=cgg-hcMAAAAJ).
The data generation was done by [Florian Haselbeck](https://bit.cs.tum.de/team/florian-haselbeck/)
The overall research was conducted by them both together with [Dominik G. Grimm](https://bit.cs.tum.de/team/dominik-grimm/) and [Christian Beecks](https://www.fernuni-hagen.de/ds/).
