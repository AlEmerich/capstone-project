(TeX-add-style-hook
 "report"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("babel" "english") ("natbib" "square" "comma" "sort" "numbers") ("ragged2e" "document") ("fontenc" "T1") ("caption" "font=scriptsize")))
   (add-to-list 'LaTeX-verbatim-environments-local "lstlisting")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "lstinline")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperref")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperimage")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperbaseurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "nolinkurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "lstinline")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
   (TeX-run-style-hooks
    "latex2e"
    "article"
    "art10"
    "babel"
    "natbib"
    "glossaries"
    "hyperref"
    "graphicx"
    "xcolor"
    "amsmath"
    "ragged2e"
    "amssymb"
    "listings"
    "multicol"
    "fontenc"
    "caption")
   (LaTeX-add-labels
    "subsub:obs_space"
    "fig:roboschoolhumanoid"
    "fig:actor-critic"
    "fig:algoDDPG"
    "fig:randompolicy"
    "fig:benchmark")
   (LaTeX-add-bibliographies)
   (LaTeX-add-xcolor-definecolors
    "codegreen"
    "codegray"
    "codepurple"
    "backcolour")
   (LaTeX-add-listings-lstdefinestyles
    "mystyle"))
 :latex)
