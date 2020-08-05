# Audio Processing Libraries

## APAV_AMV
Python library designed for my university subject *Algorismia i Programació Audiovisual*. The descriptions and information on the file are in Catalan. I can translate it if you ask me without any problem! In this library, you can find the following functions and classes:

**CLASSES**
- filtreFIR
- filtreIIR
- dft_N

**FUNCTIONS**
- lectura_Wav: reads a ".wav" audio file.
- escriptura_Wav: writes a ".wav" audio file from an array and some parameters.
- transformadaFourier: basic Fourier Transform.
- transformadaFourier_F1: returns the transforadaFourier return value at the specified frequency.
- zeros_pols: calculates the zeroes and poles of an expression.
- plantilla_modul: given the desired parameters of a filter, it returns a graphical template of its module.
- plantilla_guany: given the desired parameters of a filter, it returns a graphical template of its gain.
- representa_FIR: given its coefficients, represents the desired filter.
- fir_optim: calculates the best coefficients that define a filter from a set of given parameters.
- fft: executes the Fast Fourier Transform on an input signal with the butterfly method.

*All the intern functions used by the functions mentioned above were not included in this list.*

PD: Please, let me know if you find any mistake or have any suggestion.
