Annotations are located in the folder anno and named after the sequences names. 
For short-term tracking, we divide some long sequences into smaller segments and add them to the dataset UAV123.
If an annotation contains an underscore, it correspond to a subsection of a long video sequence.
For example, the annotations "car1_1.txt", "car1_2.txt", "car1_3.txt" all correspond the the same sequence car1. 
So in this case there is only one target but the long sequence is divided into three shorter segments. 
The sectioning can be found in the document DatasetAnnotation.pdf and in the function configSeqs.m.
You will also find the annotation for most full sequences (e.g. car1.txt) in the folder UAV20L. 
The idea to evaluate short-term tracking (UAV123) and long-term tracking (UAV20L) performance seperately. 

If you download the tracker_benchmark_v1.1 from our website, you can also have a look at the file util/configSeqs.m.
This file configures the two datasets UAV123 and UAV20L for OTB50/OTB100 and specifies all start and end frames. 

The NaN annotations mean that the object is either fully occluded or outside the frame. 

If you are interested to do the UAV123_10fps experiments as described in our paper,
you can download the downsampled dataset with corresponding annotations and configSeq.m file from our website. 

Please refer to our websites, paper and supplementary material for more details. 
For any further questions feel free to contact me at matthias.mueller.2-at-kaust.edu.sa.

Dataset Website:
https://ivul.kaust.edu.sa/Pages/Dataset-UAV123.aspx
You can downlad the datasets, annotations, benchmark tools and a document containing all annotation details here.

Project Website:
https://ivul.kaust.edu.sa/Pages/pub-benchmark-simulator-uav.aspx
 

If you use our dataset, benchmark or simulator please cite our work.
 
@Inbook{Mueller2016,
author="Mueller, Matthias
and Smith, Neil
and Ghanem, Bernard",
editor="Leibe, Bastian
and Matas, Jiri
and Sebe, Nicu
and Welling, Max",
title="A Benchmark and Simulator for UAV Tracking",
bookTitle="Computer Vision -- ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11--14, 2016, Proceedings, Part I",
year="2016",
publisher="Springer International Publishing",
address="Cham",
pages="445--461",
isbn="978-3-319-46448-0",
doi="10.1007/978-3-319-46448-0_27",
url="http://dx.doi.org/10.1007/978-3-319-46448-0_27"
}


