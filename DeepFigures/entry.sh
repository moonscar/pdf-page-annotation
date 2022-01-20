cat arxiv_list | while read line
do
    cd latex;
    sh ../latex2pdf.sh $line;
done
