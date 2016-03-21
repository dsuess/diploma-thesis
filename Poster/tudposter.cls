%% LaTeX-Klasse für Poster im Corporate Design der TU Dresden.
%%
%% Autor: Martin Zabel (martin.zabel@tu-dresden.de)
%%
%% Klassenoptionen:
%% - a0paper: DIN A0
%% - a4paper: DIN A4
%% - bluebg: Umschalten auf blauen Hintergrund im Inhaltsbereich.
%% - serifmath: Formeln mit Serifenschrift (CM/EC), statt Univers.
%%
%% TODO: Schriftgrößen und Abstände für DIN A6.
%%
%% Änderungen:
%% v0.7) - Standardmäßig Formeln wie bei tudbook.
%%       - Automatik für \@bigheadings.
%%       - Angepasste Schriftgrößen.
%% v0.6) - Bei Option bluebg: Balken korrigiert, jetzt mit weißen Linien.
%% v0.5) - Serifenlose Formeln (Deaktivierbar).
%%       - Korrekte Größe und Farben für Überschriften (section, u.a.).
%%       - Punkte für Anstriche (\textbullet) korrigiert.
%%
%% v0.4) - Optionaler blauer Hintergrund im Inhaltsbereich.
%%
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{tudposter}[2010/11/11 v0.7 martin.zabel@tu-dresden.de]

\newif\if@tudposter@bluebg\@tudposter@bluebgfalse
\DeclareOption{bluebg}{\@tudposter@bluebgtrue}

% Analog tudbook.cls
\newif\if@tudposter@sansmath\@tudposter@sansmathtrue
\DeclareOption{serifmath}{\@tudposter@sansmathfalse}

\DeclareOption*{
  % scrartcl: Für Papierformat und Basislayout, Schriftgrößen,
  % Description, ...
  \PassOptionsToClass{\CurrentOption}{scrartcl}%
  \PassOptionsToClass{\CurrentOption}{color}%
  \PassOptionsToClass{\CurrentOption}{graphicx}%
}
\ProcessOptions\relax

% Das Paket a0poster.cls konnte aufgrund eines komischen Fehlers in den
% pagestyles nicht direkt benutzt werden.
% Für zusätzliche Schriftgrößen der CM-/EC-Schriften, die nach wie vor
% für mathematische Symbole benötigt werden, wird außerdem das Paket
% fix-cm.sty statt a0size.sty verwendet. Siehe auch fixltx2e.pdf in
% der Dokumentation zu TeX-Live.
\RequirePackage{fix-cm}% Vor dem Laden einer LaTeX-Klasse!
\LoadClass{scrartcl}
%\AtBeginDvi{\special{! /landplus90 true store}}% für landscape

\usepackage{calc,color,graphicx}

% Konfiguration
% Vom Nutzer anzugebende / veraenderbare Namen und Bezeichnungen
% analog tudbeamer.cls
\newcommand*\@einrichtung{}
\newcommand*\@fachrichtung{}
\newcommand*\@institut{}
\newcommand*\@professur{}
\newcommand*\@fusszeile{}

\newcommand*{\einrichtung}[1]{\renewcommand*{\@einrichtung}{#1}}
\newcommand*{\fachrichtung}[1]{\renewcommand*{\@fachrichtung}{#1}}
\newcommand*{\institut}[1]{\renewcommand*{\@institut}{#1}}
\newcommand*{\professur}[1]{\renewcommand*{\@professur}{#1}}
\newcommand*{\fusszeile}[1]{\renewcommand*{\@fusszeile}{#1}}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Schriftart für Text
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\renewcommand*{\familydefault}{\sfdefault}
\renewcommand*{\seriesdefault}{l}
\renewcommand*{\shapedefault}{n}
\renewcommand*{\sfdefault}{aun}
\renewcommand*{\ttdefault}{aun}
\renewcommand*{\rmdefault}{aun}% Required for package: url
\normalfont

% aus tudbook.cls
\newcommand*{\dinBold}{\fontfamily{din}\fontseries{b}\selectfont}

% \textbullet aus Encoding TS1 für korrekte horizontale Ausrichtung.
% Encoding TS1 bereits durch fix-cm.sty verfügbar.
\DeclareTextSymbolDefault{\textbullet}{TS1}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Schriftart für Mathematisches
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Korrekte Symbolgrößen
\usepackage{exscale}
% cmex in beliebigen Größen, analog fix-cm.sty
\DeclareFontShape{OMX}{cmex}{m}{n}{%
      <-8>sfixed * cmex7
      <8-9> cmex8
      <9-10> cmex9
      <10->cmex10
       }{}
\DeclareSymbolFont{largesymbols}{OMX}{cmex}{m}{n}


\if@tudposter@sansmath
    % Standardmäßig werden Formeln wie bei tudbook gesetzt, um
    % Kompatibilität herzustellen.
    % - Zahlen, Buchstaben und Satzzeichen werden aus Univers entnommen.
    % - Da Univers nur in der Kodierung T1/OT1 (statt OML) vorliegt, müssen
    %   die zugehörigen Tex-Symbole neu definiert werden.
    % - Die übrigen Mathematiksymbole (z.B. Relationszeichen) werden aus
    %   den CM/EC-Schriften entnommen.
    % Hier wird ein Auszug aus tudbook.cls von Klaus Bergmann wiederverwendet.
    \DeclareSymbolFont{aunLetters}{T1}{aun}{l}{sl}
    \SetSymbolFont{aunLetters}{bold}{T1}{aun}{b}{sl}
    \DeclareSymbolFont{aunOperators}{T1}{aun}{l}{n}
    \SetSymbolFont{aunOperators}{bold}{T1}{aun}{b}{n}
    \DeclareMathSymbol{a}{\mathalpha}{aunLetters}{`a}
    \DeclareMathSymbol{b}{\mathalpha}{aunLetters}{`b}
    \DeclareMathSymbol{c}{\mathalpha}{aunLetters}{`c}
    \DeclareMathSymbol{d}{\mathalpha}{aunLetters}{`d}
    \DeclareMathSymbol{e}{\mathalpha}{aunLetters}{`e}
    \DeclareMathSymbol{f}{\mathalpha}{aunLetters}{`f}
    \DeclareMathSymbol{g}{\mathalpha}{aunLetters}{`g}
    \DeclareMathSymbol{h}{\mathalpha}{aunLetters}{`h}
    \DeclareMathSymbol{i}{\mathalpha}{aunLetters}{`i}
    \DeclareMathSymbol{j}{\mathalpha}{aunLetters}{`j}
    \DeclareMathSymbol{k}{\mathalpha}{aunLetters}{`k}
    \DeclareMathSymbol{l}{\mathalpha}{aunLetters}{`l}
    \DeclareMathSymbol{m}{\mathalpha}{aunLetters}{`m}
    \DeclareMathSymbol{n}{\mathalpha}{aunLetters}{`n}
    \DeclareMathSymbol{o}{\mathalpha}{aunLetters}{`o}
    \DeclareMathSymbol{p}{\mathalpha}{aunLetters}{`p}
    \DeclareMathSymbol{q}{\mathalpha}{aunLetters}{`q}
    \DeclareMathSymbol{r}{\mathalpha}{aunLetters}{`r}
    \DeclareMathSymbol{s}{\mathalpha}{aunLetters}{`s}
    \DeclareMathSymbol{t}{\mathalpha}{aunLetters}{`t}
    \DeclareMathSymbol{u}{\mathalpha}{aunLetters}{`u}
    \DeclareMathSymbol{v}{\mathalpha}{aunLetters}{`v}
    \DeclareMathSymbol{w}{\mathalpha}{aunLetters}{`w}
    \DeclareMathSymbol{x}{\mathalpha}{aunLetters}{`x}
    \DeclareMathSymbol{y}{\mathalpha}{aunLetters}{`y}
    \DeclareMathSymbol{z}{\mathalpha}{aunLetters}{`z}
    \DeclareMathSymbol{A}{\mathalpha}{aunLetters}{`A}
    \DeclareMathSymbol{B}{\mathalpha}{aunLetters}{`B}
    \DeclareMathSymbol{C}{\mathalpha}{aunLetters}{`C}
    \DeclareMathSymbol{D}{\mathalpha}{aunLetters}{`D}
    \DeclareMathSymbol{E}{\mathalpha}{aunLetters}{`E}
    \DeclareMathSymbol{F}{\mathalpha}{aunLetters}{`F}
    \DeclareMathSymbol{G}{\mathalpha}{aunLetters}{`G}
    \DeclareMathSymbol{H}{\mathalpha}{aunLetters}{`H}
    \DeclareMathSymbol{I}{\mathalpha}{aunLetters}{`I}
    \DeclareMathSymbol{J}{\mathalpha}{aunLetters}{`J}
    \DeclareMathSymbol{K}{\mathalpha}{aunLetters}{`K}
    \DeclareMathSymbol{L}{\mathalpha}{aunLetters}{`L}
    \DeclareMathSymbol{M}{\mathalpha}{aunLetters}{`M}
    \DeclareMathSymbol{N}{\mathalpha}{aunLetters}{`N}
    \DeclareMathSymbol{O}{\mathalpha}{aunLetters}{`O}
    \DeclareMathSymbol{P}{\mathalpha}{aunLetters}{`P}
    \DeclareMathSymbol{Q}{\mathalpha}{aunLetters}{`Q}
    \DeclareMathSymbol{R}{\mathalpha}{aunLetters}{`R}
    \DeclareMathSymbol{S}{\mathalpha}{aunLetters}{`S}
    \DeclareMathSymbol{T}{\mathalpha}{aunLetters}{`T}
    \DeclareMathSymbol{U}{\mathalpha}{aunLetters}{`U}
    \DeclareMathSymbol{V}{\mathalpha}{aunLetters}{`V}
    \DeclareMathSymbol{W}{\mathalpha}{aunLetters}{`W}
    \DeclareMathSymbol{X}{\mathalpha}{aunLetters}{`X}
    \DeclareMathSymbol{Y}{\mathalpha}{aunLetters}{`Y}
    \DeclareMathSymbol{Z}{\mathalpha}{aunLetters}{`Z}

    \DeclareMathSymbol{,}{\mathpunct}{aunLetters}{`,}
    \DeclareMathSymbol{.}{\mathord}{aunLetters}{`.}
    \DeclareMathSymbol{/}{\mathord}{aunLetters}{`/}

    \DeclareMathSymbol{0}\mathalpha{aunOperators}{"30}
    \DeclareMathSymbol{1}\mathalpha{aunOperators}{"31}
    \DeclareMathSymbol{2}\mathalpha{aunOperators}{"32}
    \DeclareMathSymbol{3}\mathalpha{aunOperators}{"33}
    \DeclareMathSymbol{4}\mathalpha{aunOperators}{"34}
    \DeclareMathSymbol{5}\mathalpha{aunOperators}{"35}
    \DeclareMathSymbol{6}\mathalpha{aunOperators}{"36}
    \DeclareMathSymbol{7}\mathalpha{aunOperators}{"37}
    \DeclareMathSymbol{8}\mathalpha{aunOperators}{"38}
    \DeclareMathSymbol{9}\mathalpha{aunOperators}{"39}

    \DeclareSymbolFontAlphabet{\mathsf}{aunOperators}
    \DeclareSymbolFontAlphabet{\mathnormal}{aunLetters}
    \DeclareMathAlphabet\mathbf{T1}{aun}{b}{n}
    \SetMathAlphabet\mathbf{bold}{T1}{aun}{b}{n}
    \DeclareMathAlphabet\mathit{T1}{aun}{l}{sl}
    \SetMathAlphabet\mathit{bold}{T1}{aun}{l}{sl}
% Ende des Auszugs    
\fi

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Schriftgrößen für einzelne Papiergrößen
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\ifdim\paperheight>842mm
%-- Gestaltungshöhe: über Höhe DIN A1 bis Höhe DIN A0

%% Copied from a0poster.sty from TeX-Live distribution.
%% Copyright (C) 1997-2004 Gerlinde Kettl and Matthias Weiser
  \renewcommand{\tiny}{\fontsize{12}{14}\selectfont}
  \renewcommand{\scriptsize}{\fontsize{14.4}{18}\selectfont}   
  \renewcommand{\footnotesize}{\fontsize{17.28}{22}\selectfont}
  \renewcommand{\small}{\fontsize{20.74}{25}\selectfont}
  \renewcommand{\normalsize}{\fontsize{24.88}{30}\selectfont}
  \renewcommand{\large}{\fontsize{29.86}{37}\selectfont}
  \renewcommand{\Large}{\fontsize{35.83}{45}\selectfont}
  \renewcommand{\LARGE}{\fontsize{43}{54}\selectfont}
  \renewcommand{\huge}{\fontsize{51.6}{64}\selectfont}
  \renewcommand{\Huge}{\fontsize{61.92}{77}\selectfont}
  \newcommand{\veryHuge}{\fontsize{74.3}{93}\selectfont}
  \newcommand{\VeryHuge}{\fontsize{89.16}{112}\selectfont}
  \newcommand{\VERYHuge}{\fontsize{107}{134}\selectfont}

  \setlength\smallskipamount{6pt plus 2pt minus 2pt}
  \setlength\medskipamount{12pt plus 4pt minus 4pt}
  \setlength\bigskipamount{24pt plus 8pt minus 8pt}

  \setlength\abovecaptionskip{25pt}
  \setlength\belowcaptionskip{0pt}
  \setlength\abovedisplayskip{25pt plus 6pt minus 15 pt}
  \setlength\abovedisplayshortskip{0pt plus 6pt}
  \setlength\belowdisplayshortskip{13pt plus 7pt minus 6pt}
  \setlength\belowdisplayskip\abovedisplayskip
%% End of Copy

\else\ifdim\paperheight>595mm
%-- Gestaltungshöhe: über Höhe DIN A2 bis Höhe DIN A1
%-- entspricht auch A0 quer
  \renewcommand{\tiny}{\fontsize\@xpt\@xiipt\selectfont}% 10pt
  \renewcommand{\scriptsize}{\fontsize{12}{14}\selectfont}
  \renewcommand{\footnotesize}{\fontsize{14.4}{18}\selectfont}   
  \renewcommand{\small}{\fontsize{17.28}{22}\selectfont}
  \renewcommand{\normalsize}{\fontsize{20.74}{25}\selectfont}% ~= 24.88/1.22 = 20.39
  \renewcommand{\large}{\fontsize{24.88}{30}\selectfont}
  \renewcommand{\Large}{\fontsize{29.86}{37}\selectfont}
  \renewcommand{\LARGE}{\fontsize{35.83}{45}\selectfont}
  \renewcommand{\huge}{\fontsize{43}{54}\selectfont}
  \renewcommand{\Huge}{\fontsize{51.6}{64}\selectfont}
  \newcommand{\veryHuge}{\fontsize{61.92}{77}\selectfont}
  \newcommand{\VeryHuge}{\fontsize{74.3}{93}\selectfont}
  \newcommand{\VERYHuge}{\fontsize{89.16}{112}\selectfont}

  \setlength\smallskipamount{5pt plus 2pt minus 2pt}
  \setlength\medskipamount{10pt plus 3pt minus 3pt}
  \setlength\bigskipamount{20pt plus 8pt minus 8pt}

  \setlength\abovecaptionskip{20pt}
  \setlength\belowcaptionskip{0pt}
  \setlength\abovedisplayskip{20pt plus 5pt minus 13 pt}
  \setlength\abovedisplayshortskip{0pt plus 5pt}
  \setlength\belowdisplayshortskip{11pt plus 6pt minus 5pt}
  \setlength\belowdisplayskip\abovedisplayskip

\else\ifdim\paperheight>421mm
%-- Gestaltungshöhe: über Höhe DIN A3 bis Höhe DIN A2
%-- entspricht auch A1 quer
  \renewcommand{\tiny}{\fontsize\@viiipt{9.5}\selectfont}% 8 pt
  \renewcommand{\scriptsize}{\fontsize\@xpt\@xiipt\selectfont}% 10pt
  \renewcommand{\footnotesize}{\fontsize{12}{14}\selectfont}
  \renewcommand{\small}{\fontsize{14.4}{18}\selectfont}   
  \renewcommand{\normalsize}{\fontsize{17.28}{22}\selectfont}% ~= 20.39/1.22 = 16.72
  \renewcommand{\large}{\fontsize{20.74}{25}\selectfont}
  \renewcommand{\Large}{\fontsize{24.88}{30}\selectfont}
  \renewcommand{\LARGE}{\fontsize{29.86}{37}\selectfont}
  \renewcommand{\huge}{\fontsize{35.83}{45}\selectfont}
  \renewcommand{\Huge}{\fontsize{43}{54}\selectfont}
  \newcommand{\veryHuge}{\fontsize{51.6}{64}\selectfont}
  \newcommand{\VeryHuge}{\fontsize{61.92}{77}\selectfont}
  \newcommand{\VERYHuge}{\fontsize{74.3}{93}\selectfont}

  \setlength\smallskipamount{4pt plus 2pt minus 2pt}
  \setlength\medskipamount{8pt plus 3pt minus 3pt}
  \setlength\bigskipamount{17pt plus 7pt minus 7pt}

  \setlength\abovecaptionskip{17pt}
  \setlength\belowcaptionskip{0pt}
  \setlength\abovedisplayskip{17pt plus 4pt minus 11 pt}
  \setlength\abovedisplayshortskip{0pt plus 4pt}
  \setlength\belowdisplayshortskip{9pt plus 5pt minus 4pt}
  \setlength\belowdisplayskip\abovedisplayskip

\else\ifdim\paperheight>298mm
%-- Gestaltungshöhe: über Höhe DIN A4 bis Höhe DIN A3

%% ala Klassenoption 12pt (size12.clo)
  \renewcommand{\tiny}{\fontsize\@vipt\@viipt\selectfont}% 6 pt
  \renewcommand{\scriptsize}{\fontsize\@viiipt{9.5}\selectfont}% 8 pt
  \renewcommand{\footnotesize}{\fontsize\@xpt\@xiipt\selectfont}% 10pt
  \renewcommand{\normalsize}{\fontsize{12}{14}\selectfont}% ~= 16.72/1.22 = 13.70
  \renewcommand{\large}{\fontsize{14.4}{18}\selectfont}
  \renewcommand{\Large}{\fontsize{17.28}{22}\selectfont}
  \renewcommand{\LARGE}{\fontsize{20.74}{25}\selectfont}
  \renewcommand{\huge}{\fontsize{24.88}{30}\selectfont}
  \renewcommand{\Huge}{\fontsize{29.86}{37}\selectfont}
  \newcommand{\veryHuge}{\fontsize{35.83}{45}\selectfont}
  \newcommand{\VeryHuge}{\fontsize{43}{54}\selectfont}
  \newcommand{\VERYHuge}{\fontsize{51.6}{64}\selectfont}

  \setlength\smallskipamount{3\p@ \@plus 1\p@ \@minus 1\p@}
  \setlength\medskipamount{6\p@ \@plus 2\p@ \@minus 2\p@}
  \setlength\bigskipamount{12\p@ \@plus 4\p@ \@minus 4\p@}

  \abovedisplayskip 12\p@ \@plus3\p@ \@minus7\p@
  \abovedisplayshortskip \z@ \@plus3\p@
  \belowdisplayshortskip 6.5\p@ \@plus3.5\p@ \@minus3\p@
  \belowdisplayskip \abovedisplayskip

\else\ifdim\paperheight>148mm
%-- Gestaltungshöhe: über Höhe DIN A6 bis Höhe DIN A4

% Schriftgrößen und Abstände werden durch scrartcl gesetzt (für 11 pt).
% 13.70/1.22 ~= 11 pt

% Zusätzliche Schriftgrößen
  \newcommand{\veryHuge}{\fontsize{29.86}{37}\selectfont}
  \newcommand{\VeryHuge}{\fontsize{35.83}{45}\selectfont}
  \newcommand{\VERYHuge}{\fontsize{43}{54}\selectfont}

\else
%-- Gestaltungshöhe: Höhe DIN A6

% TODO
\relax

\fi\fi\fi\fi\fi

\normalsize

% Für KOMAScript aus TexLive 2007 muss \@bigheadings expliziz nochmal
% aufgerufen werden
\ifdefined\@bigheadings\@bigheadings\fi

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Logo (ala tudbeamer)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\def\tudposter@logo{logo_weiss}
\ifx\pdfoutput\undefined\else\ifx\pdfoutput\relax\else\ifcase\pdfoutput\else
\def\tudposter@logo{TU_Logo_SW}\fi\fi\fi

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Seiten-Maße
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newlength{\tudposter@headheight}% Höhe des Kopfbereichs
\newlength{\tudposter@cbheight}% Höhe des Querbalkens
\newlength{\tudposter@footheight}% Fußbereich-Höhe
\newlength{\tudposter@lmargin}% Achsenabstand linker Rand
\newlength{\tudposter@rmargin}% Achsenabstand linker Rand (automatisch)
\newlength{\tudposter@logospacex}% Logo-Abstand x
\newlength{\tudposter@logospacey}% Logo-Abstand y
\newlength{\tudposter@logowidth}% Logo-Breite
\newlength{\tudposter@cbfontsize}% Schriftgröße für Querbalkentext
\newlength{\tudposter@cbtextraise}% Verschiebung für Querbalkentext

% Die Vergleichswerte sind jeweils ein Millimeter größer als nötig.
% Maße für A0 hoch und A1 hoch aus den Vorlage-Dateien
% entnommen. Diese weichen bewusst von den Angaben im CD-Handbuch
% ab. Abweichungen als Kommentar angegeben.
\ifdim\paperheight>842mm
%-- Gestaltungshöhe: über Höhe DIN A1 bis Höhe DIN A0
  \setlength{\tudposter@headheight}{140mm}%
  \setlength{\tudposter@cbheight}{20mm}%
  \setlength{\tudposter@footheight}{74mm}%
  \setlength{\tudposter@lmargin}{120mm}%
  \setlength{\tudposter@logospacex}{44mm}%
  \setlength{\tudposter@logospacey}{54mm}%
  \setlength{\tudposter@logowidth}{228mm}%
  \setlength{\tudposter@cbfontsize}{29.86pt}% Vorgabe 20pt
  \setlength{\tudposter@cbtextraise}{6mm}

\else\ifdim\paperheight>595mm
%-- Gestaltungshöhe: über Höhe DIN A2 bis Höhe DIN A1
%-- entspricht auch A0 quer
  \setlength{\tudposter@headheight}{100mm}%
  \setlength{\tudposter@cbheight}{14mm}%
  \setlength{\tudposter@footheight}{60mm}%
  \setlength{\tudposter@lmargin}{85mm}%
  \setlength{\tudposter@logospacex}{31mm}%
  \setlength{\tudposter@logospacey}{38mm}%
  \setlength{\tudposter@logowidth}{161.5mm}%
  \setlength{\tudposter@cbfontsize}{20.74pt}% Vorgabe 14pt
  \setlength{\tudposter@cbtextraise}{4mm}

\else\ifdim\paperheight>421mm
%-- Gestaltungshöhe: über Höhe DIN A3 bis Höhe DIN A2
%-- entspricht auch A1 quer
  \setlength{\tudposter@headheight}{70mm}%
  \setlength{\tudposter@cbheight}{10mm}%
  \setlength{\tudposter@footheight}{42mm}%
  \setlength{\tudposter@lmargin}{60mm}%
  \setlength{\tudposter@logospacex}{22mm}%
  \setlength{\tudposter@logospacey}{27mm}%
  \setlength{\tudposter@logowidth}{114mm}%
  \setlength{\tudposter@cbfontsize}{14.4pt}% Vorgabe 13.5pt
  \setlength{\tudposter@cbtextraise}{3mm}

\else\ifdim\paperheight>298mm
%-- Gestaltungshöhe: über Höhe DIN A4 bis Höhe DIN A3
  \setlength{\tudposter@headheight}{50mm}%
  \setlength{\tudposter@cbheight}{7mm}%
  \setlength{\tudposter@footheight}{30mm}%
  \setlength{\tudposter@lmargin}{42mm}%
  \setlength{\tudposter@logospacex}{15.5mm}%
  \setlength{\tudposter@logospacey}{19mm}%
  \setlength{\tudposter@logowidth}{80mm}%
  \setlength{\tudposter@cbfontsize}{12pt}% Vorgabe 13pt
  \setlength{\tudposter@cbtextraise}{2mm}

\else\ifdim\paperheight>148mm
%-- Gestaltungshöhe: über Höhe DIN A6 bis Höhe DIN A4
  \setlength{\tudposter@headheight}{35mm}%
  \setlength{\tudposter@cbheight}{5mm}%
  \setlength{\tudposter@footheight}{21mm}%
  \setlength{\tudposter@lmargin}{30mm}%
  \setlength{\tudposter@logospacex}{11mm}%
  \setlength{\tudposter@logospacey}{13.5mm}%
  \setlength{\tudposter@logowidth}{57mm}%
  \setlength{\tudposter@cbfontsize}{9pt}% Vorgabe 9pt
  \setlength{\tudposter@cbtextraise}{1.25mm}

\else
%-- Gestaltungshöhe: Höhe DIN A6
  \setlength{\tudposter@headheight}{17.5mm}%
  \setlength{\tudposter@cbheight}{2.5mm}%
  \setlength{\tudposter@footheight}{10.5mm}%
  \setlength{\tudposter@lmargin}{15mm}%
  \setlength{\tudposter@logospacex}{5.5mm}%
  \setlength{\tudposter@logospacey}{6.5mm}%
  \setlength{\tudposter@logowidth}{28.5mm}%
  \setlength{\tudposter@cbfontsize}{6pt}% Vorgabe 6.5pt
  \setlength{\tudposter@cbtextraise}{1mm}

\fi\fi\fi\fi\fi

% Makro für die Berechnung des Layouts, damit Seitenparameter im
% Dokument angepasst werden können.
\newcommand*\tudposter@calclayout{%
% rechter Rand: lt. M. Kaden: Achsenabstand minus zweimal Querbalken
\setlength{\tudposter@rmargin}{\tudposter@lmargin-2\tudposter@cbheight}
%
\setlength{\topmargin}{-1in}% -1in wegen globaler Verschiebung
% Kopfzeile = Kopfbereich + Querbalken
\setlength{\headheight}{\tudposter@headheight+\tudposter@cbheight}
% minimaler Leerraum unter dem Querbalken (lt. M. Kaden)
\setlength{\headsep}{2\tudposter@cbheight}
% Basislinie erster Zeile: Standardwert, ggf. im Dokument anpassen
\setlength{\topskip}{\baselineskip}
% unterer Rand: Fußbereich + minimaler Abstand (lt. M. Kaden)
\setlength{\footskip}{\tudposter@footheight+2\tudposter@cbheight}
% -1in wegen globaler Verschiebung in \topmargin
\setlength{\textheight}{\paperheight-1in-\topmargin-\headheight-\headsep-\footskip}
% -1in wegen globaler Verschiebung
\setlength{\oddsidemargin}{\tudposter@lmargin-1in}
\setlength{\evensidemargin}{\oddsidemargin}
% -2in wegen globaler Verschiebung in \oddsidemargin
\setlength{\textwidth}{\paperwidth-\tudposter@lmargin-\tudposter@rmargin}
}
\tudposter@calclayout

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Seiten-Layout
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newlength{\tudposter@logoposx}% Logo-Position X
\newlength{\tudposter@logoposy}% Logo-Position Y
\setlength{\tudposter@logoposx}{-\tudposter@lmargin+\tudposter@logospacex}
% Logo-Höhe = 0.292 * Logo-Breite
\setlength{\tudposter@logoposy}{\headheight-\tudposter@logospacey-0.292\tudposter@logowidth}

% Kopf- und Fußzeile
\newcommand*{\ps@tudposter}{%
  \renewcommand*{\@oddhead}{% Kopfzeile
    \if@tudposter@bluebg
      % Blauer Inhaltsbereich
      \color{HKS41K100}%
      \raisebox{-\paperheight+\headheight}[0mm][0mm]{%
        \makebox[0mm][r]{\rule{\tudposter@lmargin}{\paperheight}}%
        \makebox[0mm][l]{\rule{\paperwidth-\tudposter@lmargin}{\paperheight}}%
      }%
      % ... mit weißen Linien statt Balken
      \color{white}
      \makebox[0mm][r]{\rule{\tudposter@lmargin}{3pt}}%
      \makebox[0mm][l]{\rule{\paperwidth-\tudposter@lmargin}{3pt}}%
      \raisebox{\tudposter@cbheight}[0mm][0mm]{
        \makebox[0mm][r]{\rule{\tudposter@lmargin}{3pt}}%
        \makebox[0mm][l]{\rule{\paperwidth-\tudposter@lmargin}{3pt}}%
      }%
    \else
      % Weißer Inhaltsbereich, mit hellblauem Balken
      \color{HKS41K50}%
      \makebox[0mm][r]{\rule{\tudposter@lmargin}{\tudposter@cbheight}}%
      \makebox[0mm][l]{\rule{\paperwidth-\tudposter@lmargin}{\tudposter@cbheight}}%
      \color{HKS41K100}%
      \raisebox{\tudposter@cbheight}[0mm][0mm]{%
        \makebox[0mm][r]{\rule{\tudposter@lmargin}{\headheight-\tudposter@cbheight}}%
        \makebox[0mm][l]{\rule{\paperwidth-\tudposter@lmargin}{\headheight-\tudposter@cbheight}}%
      }%
    \fi
    \rlap{\raisebox{\tudposter@cbtextraise}{%
	\color{white}%
	\fontsize{\tudposter@cbfontsize}{1.2\tudposter@cbfontsize}\selectfont
	% auf leere Inhalte prüfen
        \def\tudposter@comma{}%
	\ifx\@einrichtung\@empty\else\textbf{\@einrichtung}%
	  \def\tudposter@comma{ }%
	\fi%
	\ifx\@fachrichtung\@empty\else\tudposter@comma\@fachrichtung%
	  \def\tudposter@comma{, }%
	\fi%
        \ifx\@institut\@empty\else\tudposter@comma\@institut
	  \def\tudposter@comma{, }%
	\fi%
        \ifx\@professur\@empty\else\tudposter@comma\@professur\fi%
    }}%
    \rlap{\raisebox{\tudposter@logoposy}{%
	\hspace*{\tudposter@logoposx}%
	\color{white}% pdflatex: Logo-Farbe
	\includegraphics[width=\tudposter@logowidth]{\tudposter@logo}%
    }}%
    \hfill% Damit alles links ausgerichtet wird
  }%
  \renewcommand*{\@oddfoot}{% Fußzeile
    \ifdim\tudposter@footheight>0mm
    \color{HKS41K100}%
    \makebox[0mm][r]{\rule{\tudposter@lmargin}{\tudposter@footheight}}%
    \makebox[0mm][l]{\rule{\paperwidth-\tudposter@lmargin}{\tudposter@footheight}}%
    \raisebox{.5\tudposter@footheight}{%
      \begin{minipage}{\textwidth}%
	\color{white}\small
	\@fusszeile
      \end{minipage}%
    }%
    \fi
  }%
  \let\@evenhead\@oddhead% gerade und ungerade Seite gleich
  \let\@evenfoot\@oddfoot
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Farbdefinitionen
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\definecolor{HKS41K100}{cmyk}{1.00, 0.70, 0.10, 0.50}
\definecolor{HKS41K90}{cmyk}{0.9, 0.63, 0.09, 0.45}
\definecolor{HKS41K80}{cmyk}{0.8, 0.56, 0.08, 0.40}
\definecolor{HKS41K70}{cmyk}{0.7, 0.49, 0.07, 0.35}
\definecolor{HKS41K60}{cmyk}{0.6, 0.42, 0.06, 0.30}
\definecolor{HKS41K50}{cmyk}{0.5, 0.35, 0.05, 0.25}
\definecolor{HKS41K40}{cmyk}{0.4, 0.28, 0.04, 0.20}
\definecolor{HKS41K30}{cmyk}{0.3, 0.21, 0.03, 0.15}
\definecolor{HKS41K20}{cmyk}{0.2, 0.14, 0.02, 0.10}
\definecolor{HKS41K10}{cmyk}{0.1, 0.07, 0.01, 0.05}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Text-Layout
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\parskip1ex plus 1ex
\parindent0cm

\AtBeginDocument{%
  \if@tudposter@bluebg\color{white}\fi
}

\if@tudposter@bluebg
\renewcommand*\sectfont{\normalcolor\sffamily\bfseries\color{white}}
\else
\renewcommand*\sectfont{\normalcolor\sffamily\bfseries\color{HKS41K100}}
\fi
\endinput
