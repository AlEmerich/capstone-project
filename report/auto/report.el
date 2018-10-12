(TeX-add-style-hook
 "report"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("babel" "english") ("natbib" "square" "comma" "sort" "numbers") ("ragged2e" "document") ("fontenc" "T1") ("caption" "font=scriptsize") ("geometry" "margin=0.9in")))
   (add-to-list 'LaTeX-verbatim-environments-local "lstlisting")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "nolinkurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperbaseurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperimage")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperref")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "lstinline")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "lstinline")
   (TeX-run-style-hooks
    "latex2e"
    "article"
    "art10"
    "babel"
    "natbib"
    "glossaries"
    "hyperref"
    "graphicx"
    "wrapfig"
    "subcaption"
    "xcolor"
    "amsmath"
    "ragged2e"
    "amssymb"
    "listings"
    "adjustbox"
    "multicol"
    "dirtree"
    "fontenc"
    "caption"
    "geometry")
   (LaTeX-add-labels
    "subsubsec:obs_space"
    "fig:exploratory"
    "fig:actor-critic"
    "fig:algoDDPG"
    "fig:randompolicy"
    "fig:benchmark"
    "fig:structure"
    "fig:result_pendulum"
    "subsubsec:hp_params"
    "tab:hyperparams"
    "fig:baselines"
    "fig:study_bs"
    "fig:noise_decay_factor"
    "fig:study_nsf"
    "fig:study_decay_lr"
    "fig:study_reward_scaling"
    "fig:histogram_baseline")
   (LaTeX-add-bibliographies)
   (LaTeX-add-xcolor-definecolors
    "codegreen"
    "codegray"
    "codepurple"
    "backcolour")
   (LaTeX-add-listings-lstdefinestyles
    "mystyle"))
 :latex)

