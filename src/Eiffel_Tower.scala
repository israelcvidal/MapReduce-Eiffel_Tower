import org.apache.spark.mllib.clustering.{DistributedLDAModel, LDA}
import org.apache.spark.ml.feature._
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.SparkConf
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.rdd.RDD

/**
  * Created by israelcvidal on 28/05/17.
  */
object Eiffel_Tower {

	val k = 10
	val numFeatures = 10

    def main(args: Array[String]) {

        Logger.getLogger("org").setLevel(Level.OFF)
        Logger.getLogger("akka").setLevel(Level.OFF)

        val conf = new SparkConf().setAppName("Simple Application").setMaster("local[*]")
        val spark = SparkSession
          .builder()
          .config(conf)
          .getOrCreate()


        val stopWords = spark.sparkContext.textFile("Dataset/stop-word-list.csv").flatMap(line => line.split(", ")).collect()
        val file = "Dataset/eiffel-tower-reviews.json"
        val textFile = spark.read.json(file).toDF
		val toRemove = "[]".toSet
		val textColumn = textFile.select("text").rdd.map(line => line.toString().filterNot(toRemove).toLowerCase)

		//printTopKWords(textColumn, stopWords)
		//printTopKSentences(textColumn)
		//printTopKDates(textFile)

		runLDA(textFile)

    }

	def tfidf(data: DataFrame): (IDFModel, DataFrame) = {
		val hashingTF = new HashingTF()
			.setInputCol("filtered")
			.setOutputCol("rawFeatures")
			.setNumFeatures(numFeatures)
		val featurizedData = hashingTF.transform(data)

		val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
		val idfModel = idf.fit(featurizedData)

		return (idfModel, idfModel.transform(featurizedData))
	}

	def countVectorizer(data: DataFrame): (CountVectorizerModel, DataFrame) = {
		val cvModel: CountVectorizerModel = new CountVectorizer()
		.setInputCol("filtered")
		.setOutputCol("features")
		.setVocabSize(30)
		.setMinDF(2)
		.fit(data)

		return (cvModel, cvModel.transform(data))
	}

	def runLDA(data: DataFrame): Unit = {
		val sentencesData = data.select("text").where(data.col("text").isNotNull)

		val tokenizer = new Tokenizer().setInputCol("text").setOutputCol("words")
		val remover = new StopWordsRemover().setInputCol("words").setOutputCol("filtered")

		val wordsData = tokenizer.transform(sentencesData)
		val filteredData = remover.transform(wordsData)

		val USE_TF_IDF = true

		val (model, featurizedData) = if (USE_TF_IDF) tfidf(filteredData) else countVectorizer(filteredData)

		val parsedData = featurizedData.select("features").rdd
			.map(x => Vectors.dense(x.getAs[SparseVector](0).toDense.toArray))

		val corpus = parsedData.zipWithIndex.map(_.swap).cache()

		//corpus.take(5).foreach(println)

		val ldaModel = new LDA().setK(k).run(corpus)

		val vocabulary = if (!USE_TF_IDF) model.asInstanceOf[CountVectorizerModel].vocabulary else null

		val topics = ldaModel.topicsMatrix
		for (topic <- Range(0, numFeatures)) {
			print("Topic " + topic + ":")
			for (word <- Range(0, ldaModel.vocabSize)) {
				var s = " " + topics(word, topic)
				if (!USE_TF_IDF) {
					s = " ( " + vocabulary(word) + "," + s + ")"
				}
				print(s)
			}
			println()
		}
	}

	def printTopKWords(data: RDD[String], stopWords: Array[String]): Unit = {
		val wordCount = data
			.flatMap(line => line.split(" "))
			.filter(word => !stopWords.contains(word))
			.filter(word => word.length > 2)
			.map(word => (word, 1))
			.reduceByKey(_ + _)

		val topKwords = wordCount
			.sortBy(word => word._2, ascending = false)
			.take(k)

		println("Top " + k + " words: ")
		topKwords.foreach(println)
	}

	def printTopKSentences(data: RDD[String]): Unit = {
		val sentenceCount = data
			.flatMap(line => line.toString().split("\\. |\\? |\\! |\\, |\\; "))
			.filter(sentence => sentence.split(" ").length > 1)
			.map(sentence => (sentence, 1))
			.reduceByKey(_ + _)

		val topKsentences = sentenceCount
			.sortBy(word => word._2, ascending = false)
			.take(k)

		println("Top " + k + " sentences: ")
		topKsentences.foreach(println)
	}

	def printTopKDates(data: DataFrame): Unit = {
		val timeDistribution = data.select("createdAt").rdd
			.map(line => (line.toString, 1))
			.reduceByKey(_+_)

		val topKDates = timeDistribution
			.sortBy(word => word._2, ascending = false)
			.take(k)

		println("Top " + k + " dates: ")
		topKDates.foreach(println)
	}
}
