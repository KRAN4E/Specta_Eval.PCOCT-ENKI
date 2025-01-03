SUMMARY:

This repository hosts a python program which evaluates spectra data in the context of the DATA4.ZERO-ENKI project. The project aims to image cross sections of medical microtubing via spectral data gathered by an optical device.


PROJECT:

Medical microtubes are employed in minimally invasive medical procedures and devices, thanks to their small diameter and functionality under high pressures. Internal architecture of medical microtubing varies wildly; composing multiple lumen systems which provide delivery of multiple fluids at once, reduced need for repeated venipuncture, and a potential reduction in central line-associated bloodstream infections 

Often, polymer tubing is manufactured via extrusion, which involves application of heat and pressure to a body which is stretched at constant speed and then cooled [3]. In this case, polymer pellets are melted and pressured through a tip and die which creates the desired cross-sectional shape of the tubing. This process is continuous.

Enki® microtubes are the focus of this project, composing a vast set of dimensional requirements, with diameters ranging from 0.2 to 14mm with tolerances of ±0.02mm, for a variety of microtubing and microcatheters. Currently, manufacturing checks are limited to cutting a microtube specimen and investigating the cross-section underneath a microscope, essentially taking a manual approach to inspection. This method has significant drawbacks; limited automation, meaning a mundane and repetitive inspection process, poor resolution, non-inline process, and a poor specimen, because cutting the microtube deforms the polymer across the plane. Mitigating for these factors is essential: an automated and non-invasive system must be employed. 

To solve this, a novel imaging technique has been developed, drawing on state-of-the-art optical technologies. The experimental setup composed optical devices which return spectral data at given angles around the medical tubing. 


PROGRAM:

The program, in both a .py and .ipynb format, evaluates the spectral data. Spectral data is analysed to determine material interfaces. Various visualisations are made to illustrate the gathered data, including scatter, polar, and bitmap formats. Of these, the most notable is the bitmap illustration. 

In creating the bitmap, a program connects material interface points based on promximity. Various precautions are taken to ensure that points are connected appropriately. The program then runs an automated flood fill to indicate material against air. This program has been designed to function for any and all cross-sectional geometries in this project and others like it.

The gathered data, however, had two systematic errors: non-linear light emission / observation, and non-linear wavelength-position relationship. These were mitigated for via 'raw_spectra.csv' / 'dark_spectra.csv' and 'spectra_accel.csv' respectively, each within the Adjusters folder. Note that the acceleration has been mitigated for via creation of a poynomial relationship, in some instances. 

For further explanation, please read through the produced figures in the Jupyter Notebook file. Alternatively, each DATA folder contains an 'Evaluations' subfolder which contains various similar figures.
