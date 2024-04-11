# PVOCAL (Predictions of Volitile Organic Compounds in the Atmosphere using machine Learning) Model

Hello everyone from AMS!

This model was initially created as Victor Geiser's final project for the 2023 west cohort of the NASA Student Airborne Research Program. This program is an 8 week research internship where students are able to participate dircetly in NASA Airborne Science and do hands on reseach at the University of California Irvine. 

If you are a NASA SARP intern and are interested in working and expanding upon this model within the context of SARP please message me on GitHub and I'd be happy to explain elements of this model in detail!

COMING SOON: Instructions and workflow description of using global ERA5 Meteorology data with (0.25x0.25 degree) resolution with code.
COMING SOON: Use of NASA ATom data (2016-2018) - This will allow for global coverage as opposed to haveing samples just over Southern California.

## Table of Contents

- [About](#about)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Contributing](#contributing)
- [License](#license)

## About

The purpose of this project is to explore the intersection between meteorology and atmospheric chemistry. This project is in support of NASA's Transform to Open Science (TOPS) initiative!

The PVOCAL model uses meteorological variables calculated at the endpoints of backwards trajectories generated by the NOAA HYSPLIT model as inputs to make predictions about airborne VOC concentrations
  using Sci-Kit Learn's Random Forest Regression algorithm and (experiemental) Gradient Boosting Regressor. 

## Getting Started

Download the main PVOCAL.py file and the supporting .csv file. Change the directories of the .csv on the main script and it should work to the same level of success as seen on my AMS poster! Be sure to look at 'Prerequisites' section for necessary packages.

### Prerequisites

The provided 'environment.yml' file is all the packages currently required to run the main PVOCAL.py file. 

To install the environment file run this line in an anaconda prompt window:

  conda env create -f environment.yml

### Installation

Downloading the main PVOCAL.py file is the best way to go for now! There should be plans to transition this code to a more module style of coding to put on a pypi distribution but frequent changes to the program are expected in the next few months :)

## Usage

This project is in it early stages and has some significant challenges yet, but please feel free to explore what you can with the model. It has several features, with many more to be added in the coming monthes.

## Features

Supervised machine learning workflows in both Sci-kit learn Random Forest Regression and (experimental) Gradient boosting.

Predtions of 100+ VOCs are possible with the given .csv file.

## Contributing

Given the difference in spatial resolutions between the 1 x 1 degree grids and the discrete sample locations of whole air samples. As it stands right now there is presumably a data leakage between the training and testing sets with the current standard random sklearn splitting algorithm. However, since using model data for preditions is a goal of this project, this question remains unanswered. 
  - Testing the primary workflow for HRRR and NAM model data within HYSPLIT is currently in progress!
  - And while no script exists, once a converstion script from NETCDF to Air Resources Laboratory format is developed, the ERA-5 dataset presents an attractive source of high-resolution meteorology data for use in this model

Main contributions include making the model compatibe with all VOCs in the spreadsheet and adapting code so that this process is more streamlined. 

Additionally there exists a package for Utility Based Regression in R here: https://github.com/paobranco/UBL I believe that implementing this for python and as a part of this project would be benifical for more accurate predictions while using the SMOGN algorithm.

See inline comments for other possible ways to contribute to the project!

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

All contributors for PySPLIT and SMOGN!

Special acknowledgement to Paula Branco, Luís Torgo, and Rita P. Ribeiro for writing a lot of the literature this project is based off of.

---
