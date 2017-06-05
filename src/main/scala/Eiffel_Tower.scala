import org.apache.spark.mllib.clustering.{LDA, LDAModel}
import org.apache.spark.ml.feature._
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.SparkConf
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.rdd.RDD
import com.github.vspiewak.util.SentimentAnalysisUtils.{detectSentiment, SENTIMENT_TYPE}


/**
  * Created by israelcvidal on 28/05/17.
  */
object Eiffel_Tower {

	val k = 30

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
			val titleColumn = textFile.select("title").rdd.map(line => line.toString().filterNot(toRemove).toLowerCase)



//            time{printTopKWords(textColumn, stopWords)}
//			time{printTopKSentences(textColumn)}
//			time{printTopKDates(textFile)}
//			time{runLDA(textFile, spark)}
//			time{printTopKTopics(titleColumn)}
//	        time{printTopKSentiments(textColumn)}
    }

	def countVectorizer(data: DataFrame): (CountVectorizerModel, DataFrame) = {
		val cvModel: CountVectorizerModel = new CountVectorizer()
		.setInputCol("filtered")
		.setOutputCol("features")
		.setVocabSize(200)
		.setMinTF(0.05)
		.setMinDF(0.05)
		.fit(data)

		(cvModel, cvModel.transform(data))
	}

	def runLDA(data: DataFrame, spark: SparkSession): Unit = {
		import spark.sqlContext.implicits._

		val useCol = "text"
		val sentencesData = data.select(useCol).where(data.col(useCol).isNotNull)

		val tokenizer = new Tokenizer().setInputCol(useCol).setOutputCol("words")
		val remover = new StopWordsRemover().setInputCol("words").setOutputCol("filtered")

		val filterSet = "!?,.".toSet
		val wordsData = tokenizer.transform(sentencesData)
		val filteredData = remover.transform(wordsData)

		val tooFrequentWords = filteredData.select("filtered").rdd
			.flatMap(x => x.getAs[Seq[String]](0))
			.map(x => x.filterNot(filterSet))
			.map(x => (x, 1))
			.reduceByKey(_+_)
			.sortBy(x => x._2, ascending = false)
			.map(x => x._1)
			.take(10)
			.toSeq
		val newFiltered = filteredData.rdd
			.map(x => x.getAs[Seq[String]]("filtered").map(y => y.filterNot(filterSet)).diff(tooFrequentWords))
			.toDF("filtered")

		val (model, featurizedData) = countVectorizer(newFiltered)
		val vocabulary = model.vocabulary

		val parsedData = featurizedData.select("features").rdd
			.map(x => Vectors.dense(x.getAs[SparseVector](0).toDense.toArray))

		val corpus = parsedData.zipWithIndex.map(_.swap).cache()

		val lda = new LDA().setK(k)
		val ldaModel = lda.run(corpus)

		//printLdaMatrix(ldaModel, model)

		// Describe topics.
		val topics = ldaModel.describeTopics(8)
		println("The topics described by their top-weighted terms:")
		val topicsDesc = topics.zipWithIndex.map(x => (x._2+1, x._1._1.map(x => vocabulary(x))))
		topicsDesc.foreach(x => println("Topic " + x._1 + ": " + x._2.mkString(", ")))

	}

	def printLdaMatrix(ldaModel : LDAModel, vocabulary: Array[String]): Unit ={
		val topics = ldaModel.topicsMatrix
		for (topic <- Range(0, k)) {
			print("Topic " + topic + ":")
			for (word <- Range(0, ldaModel.vocabSize)) {
				val s = " ( " + vocabulary(word) + "," + topics(word, topic) + ")"
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
			.flatMap(line => line.toString().split("\\. |\\? |\\! |\\, |\\; |\\.|\\!|\\?|\\,|\\;"))
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

	def printTopKTopics(data: RDD[String]): Unit = {
		val wordCount = data
			.map(word => (word, 1))
			.reduceByKey(_ + _)

		val topKwords = wordCount
			.sortBy(word => word._2, ascending = false)
			.take(k)

		println("Top " + k + " topics: ")
		topKwords.foreach(println)
	}

    def printTopKSentiments(data: RDD[String]): Unit = {
        val sentimentCount = data
          .flatMap(line => line.toString().split("\\. |\\? |\\! |\\, |\\; |\\.|\\!|\\?|\\,|\\;"))
          .map(sentence => (detectSentiment(sentence), 1))
//            .map(line => (detectSentiment(line), 1) )
          .reduceByKey(_ + _)

        val topKsentiments = sentimentCount
        .sortBy(word => word._2, ascending = false)
        .take(k)

        println("Top " + k + " sentiments: ")
        topKsentiments.foreach(println)
    }

	def time[R](block: => R): R = {
		val t0 = System.nanoTime()
		val result = block    // call-by-name
		val t1 = System.nanoTime()
		println("Elapsed time: " + (t1 - t0) + "ns")
		result
	}
}
