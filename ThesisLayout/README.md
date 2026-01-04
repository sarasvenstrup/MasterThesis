% **************************************************

                   UCPH Thesis

       -- A LaTeX Style for Thesis at UCPH --

           Copyright 2019 M. Roald-Arbøl

% **************************************************

## README:

This work is a derivative of the amazing *Clean Thesis* style developed by Ricardo Langner (http://cleanthesis.der-ric.de). I have adjusted it to adapt a faculty-specific layout based on a single input (the \thesisTheme command). 

I have made this for my own master thesis, and since I could find no decent UCPH templates, I decided to compile this. 

## Page layout:
One thing of interest, which you might want to change is the current page settings; UCPH default settings is two-sided printing with uneven margins & cleaning the last page so new chapters always start on the right side. This can easily be edited in thesis.tex under \documentclass by toggling "twoside" to false and disabling "openright". Furthermore, under ucphthesis.sty, find "page layout modifications": Here you disable "bindingoffset" and change "left" to 2.75. Done and dusted!



% **************************************************

## License Information:

This work may be distributed and/or modified under the
conditions of the LaTeX Project Public License, either version 1.3
of this license or (at your option) any later version.
The latest version of this license is in:
       http://www.latex-project.org/lppl.txt
and version 1.3 or later is part of all distributions of LaTeX
version 2005/12/01 or later.

This work has the LPPL maintenance status `maintained'.
The Current Maintainer of this work is M. Roald-Arbøl.
This work consists of all files listed in README.md:

---

# CONTENTS OF THE UCPH STYLE

```
ucphthesis/
|-- ucphthesis.sty              - the ucphthesis style
|-- thesis.tex                  - main document
|-- README.md                   - this file
|-- figures/                    - customised title pages
    |-- theme/
        |-- hum.pdf
        |-- jura.pdf
        |-- samf.pdf
        |-- science.pdf
        |-- sund.pdf
        |-- teo.pdf
        |-- ucph.pdf    - Beware that the UCPH logo is protected by                        a European patent (see more https://bit.ly/2HVj7ZC)
```


% **************************************************
% **************************************************


Since the bulk work was done by Ricardo Langner in compiling the Clean Thesis package, here is his README:

% **************************************************

# Clean Thesis
*-- A LaTeX Style for Thesis Documents --*

## Readme:

**Clean, Simple, Elegant**
*Clean Thesis* is a LaTeX style for thesis documents, developed for my diploma thesis (Diplomarbeit). The style can be understood as my personal compromise — a typical clean looking scientific document combined and polished with minor beautifications.

The design of this *Clean Thesis* style is inspired by user guide documents from Apple Inc.

Note: If you are looking for an exact and correct style regarding typographic rules, please have a look at the "[Classic Thesis Style](see http://www.miede.de/#classicthesis)".

**Donation = Postcard**
Based on the idea of André Miede: If you like the *Clean Thesis* style I would be very pleased about a donation in the form of a POSTCARD. You can find my address at the *Clean Thesis* [website](http://cleanthesis.der-ric.de/) as well as in the [documentation](Clean-Thesis.pdf). I am going to collect all postcards and exhibit them at the website I mentioned.

**Idea and Inspiration**
The idea of providing my customized style for thesis documents passed through my mind while writing my own thesis. Motivated and inspired by the superb "[Classic Thesis Style](see http://www.miede.de/#classicthesis)" by André Miede (thanks to André for doing a great job) I decided to collect all design and style related functionality in a separate LaTeX style and provide this style to other thesis writers.

## License Information:

README.md
Copyright 2015 R. Langner

This work may be distributed and/or modified under the
conditions of the LaTeX Project Public License, either version 1.3
of this license or (at your option) any later version.
The latest version of this license is in
  http://www.latex-project.org/lppl.txt
and version 1.3 or later is part of all distributions of LaTeX
version 2005/12/01 or later.

This work has the LPPL maintenance status `maintained'.

The Current Maintainer of this work is R. Langner.

This work consists of all files listed in MANIFEST.md.