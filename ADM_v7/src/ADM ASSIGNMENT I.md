# Advanced Data Mining Assignment I

### Abstract
Short text have become highly popular, specially in the last few years due to the emergence of social media. With its growth , the mining of such short texts have also become highly important. However there are challenges in mining short texts. The text mining standards that are generally used, don't really apply to short text. Short texts have very less amount of observations to be able to catagorize or classify it accurately. This data has to be cleaned then expanded and only then can correct classification be done. In this project we have implemented the research paper [Concept Based Short Text Stream Classification][peiLi] to classify short texts. We have made changes to the model and made it anytime. In this report we will be explaining the model and our changes.

### Problem Formulation
Given a short text stream D, we can divide it into N data chucks, denoted as D = {D 1 , D 2 , . . . , D N}(N →∝), where each data chunk consists of |D i | short documents,denoted as D i = {d 1 , d 2 , . . . , d |D i | } (1 ≤ i ≤ N), and each document can be vectorized as d j = {(v j , y j )|Vj ∈ E, y∈ Y}(1 ≤j ≤ |D i |), where E = R^M indicates the domain of the featurespace, Y indicates the set of document classes with L labels. Our goal in this project is to label a datachunk to one of the L classes.

### Feature Extension
Probably the biggest difficulty in classifying short texts is the limitation of data. With very low feature space texts become very hard to classify, especially because words can mean different things based on the context. Thus for short text it becomes very important to understand the *concept* of the short text/document. To extend our feature space we use Micosoft Research's Database called *Probase*. We are using *Probase* as a ISA knowledge-base to get top-n concepts for entities. Probase gives a P(c|e) value along with each concept for the entity. A higher P(c|e) value signifies that the entity e is highly corelated to the concept c. In our project we are taking the top-5 concepts of entities based on their P(c|e) values.

### Disambiguation
Terms generally have different meaning when used in different context. An example given in the paper is the sentence *"Apple had agreed to license certain parts of its GUI to Microsoft for use in Windows 1.0"*. Here *Apple* is not a fruit but a company. However the Concept *fruit* will have a higher P(c|e) value. ie. P(c=*fruit*|*Apple*) > P(c=*company*|*Apple*). Since we are classifying the document based on all the total concepts, other terms with similar concepts increase the concepts weight for the document, leading to correct classification.

### Concept Clustering
After concept extraction for each document, we have two sets that define a document, a set of entities and a set of concepts. The weights of the entities are the tf IDF values for the document in the data chunk. The weights for the concepts are the P(c|e) values gathered from probase.
We now cluster the concepts of each document. In our project we are using KMeans++ to cluster the concepts.

### Document Clustering
Our next gloal is to cluster documents in a data chunk. A document is a set of concept clusters, consider them micro-clusters. The documents who have more concept-clusters in common will be pooled into a cluster, called a document cluster or a context.

### Training
We pass data chunks of about 2000 documents each to the training module. Currently, we have clustered up  8 data chunks belonging to one of business, computers, culture, engineering, education, health, sports and politics. It is expected to have documents that can relate to multiple topics. We consider the most dominant and outstanding according to the likelihood matching of weights as described in the paper.

##### [ShortTextTrainer.py is responsible for training.]

### Testing
We classify a data chunk to a topic based on maximum inclination of contained documents. Each data chunk goes through same processing and document clustering. A data chunk is matched to a topic based on affinity between document clusters of test data chunk and those of training data set.

##### [The ResultCollector.sh will run the test simulations, parameters need to be set in TextClassifier.py]

### Anytime implementation
We assign a fixed time to scan a data chunk for documents simulating the inter-chunk arrival time for a data chunk stream. In the traditional implementation, all the documents are scanned while in this implementation, only some docs are scanned in the fixed time interval.

### Results
The Traditional approach that took about 500 second time interval for scanning documents served with a 65-70% accuracy of document classification, about 5% were guessed and 20-30% were wrongly classified. Due to constraint on time, only 4 such simulations could be carried out and averaged.
The anytime implementation gives :
100 second time interval : about 57% accuracy of document classification, about 4% are guessed and 38% are wrongly classified. (5 simulations averaged)
90 second time interval : about 53% accuracy of document classification, about 5% are guessed and 42% are wrongly classified. (5 simulations averaged)
80 second time interval : about 46% accuracy of document classification, about 7% are guessed and 48% are wrongly classified. (5 simulations averaged)
70 second time interval : about 32% accuracy of document classification, about 12% are guessed and 57% are wrongly classified. (5 simulations averaged)

##### [The resultAnalyzer.py will collect the results and put it in analysis.txt for your reference]

[peiLi]: <https://www.ncbi.nlm.nih.gov/pubmed/28922135>