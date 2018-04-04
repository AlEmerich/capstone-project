(TeX-add-style-hook
 "proposal"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("babel" "english") ("natbib" "square" "comma" "sort" "numbers") ("fontenc" "T1")))
   (TeX-run-style-hooks
    "latex2e"
    "glossary"
    "article"
    "art10"
    "babel"
    "natbib"
    "glossaries"
    "hyperref"
    "graphicx"
    "xcolor"
    "amsmath"
    "multicol"
    "fontenc")
   (LaTeX-add-bibliographies))
 :latex)

