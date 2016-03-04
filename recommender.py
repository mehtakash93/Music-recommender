# # Music Recommender System using Apache Spark and Python
# 
# ## Description
# 
# For this project, I created a recommender system that would recommend new musical artists to a user based on their listening history. Suggesting different songs or musical artists to a user is important to many music streaming services, such as Pandora and Spotify. In addition, this type of recommender system could also be used as a means of suggesting TV shows or movies to a user (e.g., Netflix). 
# 
# To create this system I used Spark and the collaborative filtering technique. 
# **Submission Instructions:** 
 
# ## Datasets
# 
# I have used publicly available song data from audioscrobbler, which can be found [here](http://www-etud.iro.umontreal.ca/~bergstrj/audioscrobbler_data.html). However, I modified the original data files so that the code would run in a reasonable time on a single machine.


# ## Necessary Package Imports



from pyspark.mllib.recommendation import *
import random
from operator import *


# ## Loading data
# Loading the three datasets into RDDs and name them `artistData`, `artistAlias`, and `userArtistData`. 



artistData=sc.textFile("artist_data_small.txt")
artistAlias=sc.textFile("artist_alias_small.txt")
userArtistData=sc.textFile("user_artist_data_small.txt")
userArtistData.count()
userArtistData1=userArtistData.map(lambda line:line.split(" ")).map(lambda r: (int(r[0]), int(r[1]),int(r[2])))
artistAlias1=artistAlias.map(lambda line:line.split("\t")).map(lambda r: (int(r[0]), int(r[1])))
artistData1=artistData.map(lambda line:line.split("\t")).map(lambda r: (int(r[0]), r[1]))




# ## Data Exploration
# 


mean=userArtistData1.map(lambda r: (int(r[0]), int(r[1]))).groupByKey().map(lambda l:(l[0],len(list(l[1]))))
aggr=userArtistData1.map(lambda r: (int(r[0]), int(r[2]))).reduceByKey(lambda a,b:a+b)


topThree=aggr.takeOrdered(3, key=lambda x: -x[1])
topThree1=sc.parallelize(topThree)
a=topThree1.join(mean).map(lambda x: (x[0],x[1][0],x[1][0]/x[1][1]))
my_list=a.collect()
for list_elems in my_list:
    print("User "+str(list_elems[0])+" has a total play count of "+str(list_elems[1])+" and a mean play count of "+str(list_elems[2]))
  


# ####  Splitting Data for Testing


trainData,validationData,testData=userArtistData1.randomSplit([0.4,0.4,0.2],13)
trainData.cache()
validationData.cache()
testData.cache()


# ## The Recommender Model
 
# ### Model Evaluation
# 
# Although there may be several ways to evaluate a model, I have used a simple method here. Suppose we have a model and some dataset of *true* artist plays for a set of users. This model can be used to predict the top X artist recommendations for a user and these recommendations can be compared the artists that the user actually listened to (here, X will be the number of artists in the dataset of *true* artist plays). Then, the fraction of overlap between the top X predictions of the model and the X artists that the user actually listened to can be calculated. This process can be repeated for all users and an average value returned.

# The function `modelEval` will take a model (the output of ALS.trainImplicit) and a dataset as input. Parameter Tuning is done one the validation data.After parameter tuning, the model can be evaluated on the test data.



def modelEval(bestModel,Data1, trainData):
    a=trainData.map(lambda x: ((x[0]),(x[1]))).groupByKey().map(lambda r: (int(r[0]), list(r[1]))).collect()
    object_dict = dict((x[0], x[1]) for x in a)
    t=Data1.map(lambda x: ((x[0]),(x[1]))).groupByKey().map(lambda r: (int(r[0]), list(r[1]))).collect()
    object_dict1 = dict((x[0], x[1]) for x in t)
    allArtists=userArtistData1.map(lambda x: x[1]).distinct()
    t1=Data1.map(lambda g:(g[0], g[1]))
    unique_test=Data1.map(lambda x: x[0]).distinct().collect()
    sum=0
    for users in unique_test:
        userEval=[]
        nonTrainArtists=set(allArtists.collect())-set(object_dict[users])
        for art in nonTrainArtists:
            userEval.append((users,art))
        userEval=sc.parallelize(userEval)    
        trueArtist=object_dict1[users]
        mod=bestModel.predictAll(userEval)
        predictResult=mod.map(lambda l: (l[1],l[2])).takeOrdered(len(trueArtist), key=lambda x: -x[1]) 
        predictResult1=sc.parallelize(predictResult)
        predictResult1=predictResult1.map(lambda f:f[0]).collect()
        h=set(predictResult1) & set(trueArtist)
        d=len(h)/float(len(predictResult1))
        sum=sum+d
    return float(sum/float(len(unique_test)))       


# ### Model Construction



vals=[1:100]
for val in vals:
    expModel = ALS.trainImplicit(trainData, rank=val, seed=345)
    score=modelEval(expModel, validationData, trainData)
    print "The model score for rank "+str(val)+" is",'%.5f' % score




bestModel = ALS.trainImplicit(trainData, rank=10, seed=345)
modelEval(bestModel, testData, trainData)


# ## Trying Some Artist Recommendations
# Using the best model above, predicting the top 5 artists for user `1059637` using the [recommendProducts](http://spark.apache.org/docs/1.5.2/api/python/pyspark.mllib.html#pyspark.mllib.recommendation.MatrixFactorizationModel.recommendProducts) function.

x=bestModel.recommendProducts(1059637,5)
recomendations=sc.parallelize(x)
recomendationArtists=recomendations.map(lambda r:r[1]).collect()
y=artistAlias1.map(lambda x: ((x[0]),(x[1]))).groupByKey().map(lambda r: (int(r[0]), list(r[1]))).collect()
y_dict = dict((x[0], x[1]) for x in y)
realArt=artistData1.map(lambda x: ((x[0]),(x[1]))).groupByKey().map(lambda r: (int(r[0]), list(r[1]))).collect()
realArt_dict = dict((x[0], x[1]) for x in realArt)
i=0
for art in recomendationArtists:
    if art in realArt_dict.keys():
        i=i+1
        print "Artist %d :"%i+ realArt_dict[art][0]
    else:
        alias=y_dict[art]
        i=i+1
        print "Artist %d :"%i+ realArt_dict[alias][0]
        
    


