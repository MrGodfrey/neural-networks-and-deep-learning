\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{nndlcourse}[2026/03/29 Shared class for NNDL course notes]

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions\relax
\LoadClass{article}

\RequirePackage{ctex}
\RequirePackage{amsmath}
\RequirePackage{amsfonts}
\RequirePackage{graphicx}
\RequirePackage{setspace}
\RequirePackage{amssymb}
\RequirePackage{latexsym}
\RequirePackage{amsthm}
\RequirePackage{euscript}
\RequirePackage{makeidx}
\RequirePackage{color}
\RequirePackage{mathrsfs}
\RequirePackage[numbers,sort&compress]{natbib}
\RequirePackage{hyperref}
\hypersetup{
    colorlinks=true,
    linkcolor=blue,
    urlcolor=cyan,
    citecolor=cyan
}

\RequirePackage{geometry}
\geometry{
    a4paper,
    left=2.5cm,
    right=2.5cm,
    top=3cm,
    bottom=2cm
}

\newtheorem{theorem}{Theorem}[section]
\newtheorem{lemma}[theorem]{{\bf Lemma}}
\newtheorem{corollary}[theorem]{{\bf Corollary}}
\newtheorem{proposition}[theorem]{{\bf Proposition}}
\theoremstyle{remark}
\newtheorem{remark}{{\bf Remark}}[section]
\newtheorem{conclusion}[remark]{Conclusion}
\newtheorem{condition}{{\bf Condition}}[section]
\theoremstyle{definition}
\newtheorem{definition}{{\bf Definition}}[section]
\newtheorem{example}{{\bf Example}}[section]
\theoremstyle{plain}
\numberwithin{equation}{section}
\newtheorem{assumption}{Assumption}
\numberwithin{assumption}{section}

\newcommand{\deta}{\delta}

\allowdisplaybreaks

\endinput