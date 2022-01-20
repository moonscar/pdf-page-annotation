set -v

ARXIV_ID=$1
OUTPUT_DIR="$ARXIV_ID/"
mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR"
curl "https://arxiv.org/e-print/$ARXIV_ID" | tar xz

echo "Downloaded $ARXIV_ID into $OUTPUT_DIR"

TARGET_PATH=.
OUTPUT_PREFIX=.

ENTRY=`grep -l "documentclass" $TARGET_PATH/*.tex`

# COLOR_STR="\usepackage{color}
# \usepackage{floatrow}
# \usepackage{tcolorbox}
# \DeclareColorBox{figurecolorbox}{\fcolorbox{red}{white}}
# \DeclareColorBox{tablecolorbox}{\fcolorbox{yellow}{white}}
# \floatsetup[figure]{framestyle=colorbox, colorframeset=figurecolorbox, framearound=all, frameset={\fboxrule1pt\fboxsep0pt}}
# \floatsetup[table]{framestyle=colorbox, colorframeset=tablecolorbox, framearound=all, frameset={\fboxrule1pt\fboxsep0pt}}
# \usepackage[labelfont={color=green},textfont={color=blue}]{caption}
# \begin{document}"

# BLACK_STR="\usepackage{color}
# \usepackage{floatrow}
# \usepackage{tcolorbox}
# \DeclareColorBox{figurecolorbox}{\fcolorbox{white}{white}}
# \DeclareColorBox{tablecolorbox}{\fcolorbox{white}{white}}
# \floatsetup[figure]{framestyle=colorbox, colorframeset=figurecolorbox, framearound=all, frameset={\fboxrule1pt\fboxsep0pt}}
# \floatsetup[table]{framestyle=colorbox, colorframeset=tablecolorbox, framearound=all, frameset={\fboxrule1pt\fboxsep0pt}}
# \usepackage[labelfont={color=black},textfont={color=black}]{caption}
# \begin{document}"

# BEGIN_DOC = "\begin{document}"

COLOR_STR="\\\usepackage{color}\n\\\usepackage{floatrow}\n\\\usepackage{tcolorbox}\n\\\DeclareColorBox{figurecolorbox}{\\\fcolorbox{red}{white}}\n\\\DeclareColorBox{tablecolorbox}{\\\fcolorbox{yellow}{white}}\n\\\floatsetup[figure]{framestyle=colorbox, colorframeset=figurecolorbox, framearound=all, frameset={\\\fboxrule1pt\\\fboxsep0pt}}\n\\\floatsetup[table]{framestyle=colorbox, colorframeset=tablecolorbox, framearound=all, frameset={\\\fboxrule1pt\\\fboxsep0pt}}\n\\\usepackage[labelfont={color=green},textfont={color=blue}]{caption}\n\\\begin{document}"

BLACK_STR="\\\usepackage{color}\n\\\usepackage{floatrow}\n\\\usepackage{tcolorbox}\n\\\DeclareColorBox{figurecolorbox}{\\\fcolorbox{white}{white}}\n\\\DeclareColorBox{tablecolorbox}{\\\fcolorbox{white}{white}}\n\\\floatsetup[figure]{framestyle=colorbox, colorframeset=figurecolorbox, framearound=all, frameset={\\\fboxrule1pt\\\fboxsep0pt}}\n\\\floatsetup[table]{framestyle=colorbox, colorframeset=tablecolorbox, framearound=all, frameset={\\\fboxrule1pt\\\fboxsep0pt}}\n\\\usepackage[labelfont={color=black},textfont={color=black}]{caption}\n\\\begin{document}"

BEGIN_DOC="\\\begin{document}"

sed "s/$BEGIN_DOC/$COLOR_STR/g" $ENTRY > color.tex
sed "s/$BEGIN_DOC/$BLACK_STR/g" $ENTRY > black.tex

pdflatex -interaction=nonstopmode -shell-escape -output-directory ./ -output-format pdf black.tex

pdflatex -interaction=nonstopmode -shell-escape -output-directory ./ -output-format pdf color.tex
